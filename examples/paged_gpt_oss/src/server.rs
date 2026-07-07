//! OpenAI-compatible HTTP server around the continuous-batching [`Engine`].
//!
//! The engine is single-threaded (owns the CUDA graph/runtime), so it runs on a
//! dedicated OS thread and the async axum handlers talk to it over channels:
//! a handler submits a [`Request`] plus a per-request token channel; the engine
//! loop routes each sampled token back to the right channel and closes it when
//! the sequence finishes. Supports `/health`, `/v1/models`, and
//! `/v1/completions` (streaming SSE + non-streaming), which is all
//! `benchmark_serving.py --backend openai` needs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::State,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use luminal_cuda_lite::cudarc::driver::CudaStream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::engine::{Engine, EngineConfig};
use crate::scheduler::{FinishReason, Request, SamplingParams};

// ─────────────────────────── engine thread ───────────────────────────

pub enum TokenMsg {
    Token(u32),
    Done(FinishReason),
}

struct Submit {
    request: Request,
    resp: mpsc::UnboundedSender<TokenMsg>,
}

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::UnboundedSender<Submit>,
    next_id: Arc<AtomicU64>,
    /// Set true once the model has finished loading on the engine thread.
    ready: Arc<AtomicBool>,
}

impl EngineHandle {
    /// Submit a tokenized request; returns a receiver streaming its tokens.
    pub fn submit(
        &self,
        prompt: Vec<i32>,
        params: SamplingParams,
    ) -> mpsc::UnboundedReceiver<TokenMsg> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (resp, rx) = mpsc::unbounded_channel();
        let _ = self.tx.send(Submit {
            request: Request { id, prompt, params },
            resp,
        });
        rx
    }
}

/// Spawn the engine on a dedicated thread and return a handle to it.
pub fn spawn_engine(
    stream: Arc<CudaStream>,
    shards: Vec<PathBuf>,
    cfg: EngineConfig,
) -> EngineHandle {
    let (tx, mut rx) = mpsc::unbounded_channel::<Submit>();
    let next_id = Arc::new(AtomicU64::new(1));
    let ready = Arc::new(AtomicBool::new(false));
    let ready_thread = ready.clone();
    std::thread::Builder::new()
        .name("gptoss-engine".into())
        .spawn(move || {
            let mut engine = Engine::load(stream, &shards, cfg);
            ready_thread.store(true, Ordering::Release);
            let mut senders: HashMap<u64, mpsc::UnboundedSender<TokenMsg>> = HashMap::new();
            loop {
                if engine.has_work() {
                    // Drain any newly-arrived submissions without blocking.
                    while let Ok(sub) = rx.try_recv() {
                        senders.insert(sub.request.id, sub.resp);
                        engine.add_request(sub.request);
                    }
                    let out = engine.step();
                    for (id, tok) in out.emitted {
                        if let Some(s) = senders.get(&id) {
                            let _ = s.send(TokenMsg::Token(tok));
                        }
                    }
                    for fin in out.finished {
                        if let Some(s) = senders.remove(&fin.id) {
                            let _ = s.send(TokenMsg::Done(fin.reason));
                        }
                    }
                } else {
                    // Idle: block until the next submission (or shutdown).
                    match rx.blocking_recv() {
                        Some(sub) => {
                            senders.insert(sub.request.id, sub.resp);
                            engine.add_request(sub.request);
                        }
                        None => break, // all handles dropped
                    }
                }
            }
        })
        .expect("spawn engine thread");
    EngineHandle { tx, next_id, ready }
}

// ─────────────────────────── HTTP layer ───────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub model: String,
    pub default_max_tokens: usize,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

async fn health(State(s): State<AppState>) -> Response {
    if s.engine.ready.load(Ordering::Acquire) {
        (axum::http::StatusCode::OK, "OK").into_response()
    } else {
        (axum::http::StatusCode::SERVICE_UNAVAILABLE, "loading").into_response()
    }
}

async fn models(State(s): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": s.model, "object": "model", "owned_by": "luminal" }],
    }))
}

/// OpenAI `prompt` is either a string or an array of token ids (or arrays of
/// either). We accept the single-string and single-token-array forms.
#[derive(Deserialize)]
#[serde(untagged)]
enum PromptField {
    Text(String),
    Tokens(Vec<i32>),
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: PromptField,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    /// vLLM extension used by benchmark_serving to force full-length outputs.
    #[serde(default)]
    ignore_eos: bool,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn finish_str(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Rejected => "length",
    }
}

async fn completions(State(s): State<AppState>, Json(req): Json<CompletionRequest>) -> Response {
    let prompt_ids: Vec<i32> = match req.prompt {
        PromptField::Tokens(t) => t,
        PromptField::Text(text) => match s.tokenizer.encode(text, false) {
            Ok(enc) => enc.get_ids().iter().map(|&x| x as i32).collect(),
            Err(e) => {
                return (
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("tokenize error: {e}"),
                )
                    .into_response();
            }
        },
    };
    let prompt_tokens = prompt_ids.len();
    let params = SamplingParams {
        max_tokens: req.max_tokens.unwrap_or(s.default_max_tokens).max(1),
        ignore_eos: req.ignore_eos,
    };
    let rx = s.engine.submit(prompt_ids, params);
    let id = format!("cmpl-{}", now_secs());
    let created = now_secs();
    let model = s.model.clone();
    let tok = s.tokenizer.clone();

    if req.stream {
        stream_completion(rx, id, created, model, tok, prompt_tokens).into_response()
    } else {
        full_completion(rx, id, created, model, tok, prompt_tokens)
            .await
            .into_response()
    }
}

/// Decode the running id list and return the newly-appended text suffix.
fn decode_delta(tok: &Tokenizer, ids: &[u32], prev_text_len: usize) -> (String, usize) {
    let full = tok.decode(ids, false).unwrap_or_default();
    let delta = full.get(prev_text_len..).unwrap_or("").to_string();
    (delta, full.len())
}

fn stream_completion(
    mut rx: mpsc::UnboundedReceiver<TokenMsg>,
    id: String,
    created: u64,
    model: String,
    tok: Arc<Tokenizer>,
    prompt_tokens: usize,
) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let stream = async_stream::stream! {
        let mut ids: Vec<u32> = Vec::new();
        let mut text_len = 0usize;
        while let Some(msg) = rx.recv().await {
            match msg {
                TokenMsg::Token(t) => {
                    ids.push(t);
                    let (delta, new_len) = decode_delta(&tok, &ids, text_len);
                    text_len = new_len;
                    let chunk = json!({
                        "id": id, "object": "text_completion", "created": created,
                        "model": model,
                        "choices": [{ "index": 0, "text": delta, "finish_reason": serde_json::Value::Null }],
                    });
                    yield Ok(Event::default().data(chunk.to_string()));
                }
                TokenMsg::Done(reason) => {
                    let chunk = json!({
                        "id": id, "object": "text_completion", "created": created,
                        "model": model,
                        "choices": [{ "index": 0, "text": "", "finish_reason": finish_str(reason) }],
                    });
                    yield Ok(Event::default().data(chunk.to_string()));
                    break;
                }
            }
        }
        // Final usage chunk (OpenAI stream_options.include_usage) — benchmark
        // tools read completion_tokens from here for throughput/TPOT.
        let usage = json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": ids.len(),
                "total_tokens": prompt_tokens + ids.len(),
            },
        });
        yield Ok(Event::default().data(usage.to_string()));
        yield Ok(Event::default().data("[DONE]"));
    };
    // NOTE: no SSE keep-alive. axum's default keep-alive injects a `:` comment
    // line every 15s of silence, but the InferenceX/vLLM benchmark client parses
    // every non-empty SSE line as `data: <json>` and chokes on comment lines. At
    // concurrency >=2, first-token latency can exceed 15s (shared chunked
    // prefill), so the heartbeat fires before the first token and fails the
    // request. Dropping keep-alive keeps the stream benchmark-compatible.
    Sse::new(stream)
}

async fn full_completion(
    mut rx: mpsc::UnboundedReceiver<TokenMsg>,
    id: String,
    created: u64,
    model: String,
    tok: Arc<Tokenizer>,
    prompt_tokens: usize,
) -> Json<serde_json::Value> {
    let mut ids: Vec<u32> = Vec::new();
    let mut reason = "length";
    while let Some(msg) = rx.recv().await {
        match msg {
            TokenMsg::Token(t) => ids.push(t),
            TokenMsg::Done(r) => {
                reason = finish_str(r);
                break;
            }
        }
    }
    let text = tok.decode(&ids, false).unwrap_or_default();
    let completion_tokens = ids.len();
    Json(json!({
        "id": id, "object": "text_completion", "created": created, "model": model,
        "choices": [{ "index": 0, "text": text, "finish_reason": reason, "logprobs": serde_json::Value::Null }],
        "usage": Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}
