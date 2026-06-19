//! Continuous-batching scheduler — pure host logic (no GPU), unit-testable.
//!
//! Every model step is **uniform**: each active sequence contributes exactly
//! one query token — a prompt token while it is still prefilling, otherwise its
//! last sampled token. So `total_s == #active sequences` (bounded by
//! `max_batch`), which keeps the intermediate arena the same size as the
//! validated demo path. (Whole-prompt prefill at large `s` blows the arena up
//! beside the 63 GB of weights; token-by-token prefill avoids that and also
//! removes any prompt-length cap.) The scheduler tracks per-seq KV slots via
//! [`KvAllocator`] and retires sequences on EOS / max_tokens.

use std::collections::VecDeque;

use crate::kv_alloc::{KvAllocator, SeqId};

/// gpt-oss harmony end tokens: `<|return|>` and `<|end|>`.
pub const EOS_TOKENS: [u32; 2] = [200002, 200007];

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            ignore_eos: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<i32>,
    pub params: SamplingParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit an EOS token.
    Stop,
    /// Hit max_tokens (or was dropped on KV exhaustion).
    Length,
    /// Empty prompt — could not be served.
    Rejected,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Finished {
    pub id: u64,
    pub reason: FinishReason,
}

struct Running {
    id: u64,
    seq: SeqId,
    params: SamplingParams,
    prompt: Vec<i32>,
    /// Next absolute position to fill (== current context length). While
    /// `pos < prompt.len()` the sequence is still prefilling.
    pos: usize,
    generated: usize,
    /// Last sampled token, fed once `pos >= prompt.len()`.
    next_token: i32,
}

/// One model step's worth of work, in batch-row order (one query per sequence).
pub struct StepPlan {
    /// `(seq, [position])` for `build_batch`.
    pub entries: Vec<(SeqId, Vec<usize>)>,
    /// Token id per row (length == #sequences), in `entries` order.
    pub tokens: Vec<i32>,
    /// `(global_row, seq)` — each sequence's single sample row.
    pub samples: Vec<(usize, SeqId)>,
}

pub struct SchedulerConfig {
    /// Max concurrent running sequences (== max total_s per step).
    pub max_batch: usize,
    /// Unused (token-by-token prefill has no prompt-length cap); kept for API
    /// stability with the engine config.
    pub max_prefill: usize,
}

pub struct Scheduler {
    alloc: KvAllocator,
    pending: VecDeque<Request>,
    running: Vec<Running>,
    next_seq: SeqId,
    cfg: SchedulerConfig,
}

impl Scheduler {
    pub fn new(kv_capacity: usize, cfg: SchedulerConfig) -> Self {
        Self {
            alloc: KvAllocator::new(kv_capacity),
            pending: VecDeque::new(),
            running: Vec::new(),
            next_seq: 0,
            cfg,
        }
    }

    pub fn add_request(&mut self, req: Request) {
        self.pending.push_back(req);
    }

    pub fn has_work(&self) -> bool {
        !self.pending.is_empty() || !self.running.is_empty()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn kv_used(&self) -> usize {
        self.alloc.used_slots()
    }

    /// Read-only view of the slot allocator (for `build_batch`). Reflects the
    /// slots allocated by the most recent [`Scheduler::schedule`].
    pub fn allocator(&self) -> &KvAllocator {
        &self.alloc
    }

    /// Build the next uniform step: admit pending requests up to `max_batch`
    /// (KV permitting), then have every running sequence emit one query token.
    /// Returns `(plan, finished)`; `plan` is `None` when there is nothing to do.
    pub fn schedule(&mut self) -> (Option<StepPlan>, Vec<Finished>) {
        let mut finished = Vec::new();

        // Admit pending requests while there's batch room and KV headroom for
        // this step (one new slot per sequence, new and existing).
        while self.running.len() < self.cfg.max_batch {
            let Some(req) = self.pending.front() else { break };
            if req.prompt.is_empty() {
                let req = self.pending.pop_front().unwrap();
                finished.push(Finished {
                    id: req.id,
                    reason: FinishReason::Rejected,
                });
                continue;
            }
            if !self.alloc.can_allocate(self.running.len() + 1) {
                break; // not enough KV for all sequences' next slot this step
            }
            let req = self.pending.pop_front().unwrap();
            let seq = self.next_seq;
            self.next_seq += 1;
            self.running.push(Running {
                id: req.id,
                seq,
                params: req.params,
                prompt: req.prompt,
                pos: 0,
                generated: 0,
                next_token: 0,
            });
        }

        let mut entries: Vec<(SeqId, Vec<usize>)> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();
        let mut samples: Vec<(usize, SeqId)> = Vec::new();
        let mut row = 0usize;
        let mut drop_seqs: Vec<SeqId> = Vec::new();

        for i in 0..self.running.len() {
            let (id, seq, pos, tok) = {
                let r = &self.running[i];
                let tok = if r.pos < r.prompt.len() {
                    r.prompt[r.pos] // still prefilling: feed the next prompt token
                } else {
                    r.next_token // decoding: feed the last sampled token
                };
                (r.id, r.seq, r.pos, tok)
            };
            // Allocate the slot for this step's new position.
            if self.alloc.allocate(seq, 1).is_none() {
                finished.push(Finished {
                    id,
                    reason: FinishReason::Length,
                });
                drop_seqs.push(seq);
                continue;
            }
            entries.push((seq, vec![pos]));
            tokens.push(tok);
            samples.push((row, seq));
            row += 1;
        }
        for seq in drop_seqs {
            self.alloc.free(seq);
            self.running.retain(|r| r.seq != seq);
        }

        if entries.is_empty() {
            return (None, finished);
        }
        (Some(StepPlan { entries, tokens, samples }), finished)
    }

    /// Feed back the sampled token for each sequence. Outputs sampled while a
    /// sequence is still prefilling (before its last prompt token) are
    /// intermediate and discarded; the rest are emitted and checked against
    /// EOS / max_tokens. Returns `(emitted, finished)`.
    pub fn ingest(&mut self, sampled: &[(SeqId, u32)]) -> (Vec<(u64, u32)>, Vec<Finished>) {
        let mut emitted = Vec::new();
        let mut finished = Vec::new();
        let mut free_seqs = Vec::new();

        for &(seq, tok) in sampled {
            let Some(r) = self.running.iter_mut().find(|r| r.seq == seq) else {
                continue;
            };
            // `r.pos` is the position we just fed. The sampled token is a real
            // output iff that was the last prompt token or a decode token.
            let is_output = r.pos + 1 >= r.prompt.len().max(1);
            r.pos += 1;
            if !is_output {
                continue; // prefill intermediate
            }
            r.next_token = tok as i32;
            r.generated += 1;
            emitted.push((r.id, tok));

            let is_eos = !r.params.ignore_eos && EOS_TOKENS.contains(&tok);
            if is_eos || r.generated >= r.params.max_tokens {
                finished.push(Finished {
                    id: r.id,
                    reason: if is_eos {
                        FinishReason::Stop
                    } else {
                        FinishReason::Length
                    },
                });
                free_seqs.push(seq);
            }
        }

        for seq in free_seqs {
            self.alloc.free(seq);
            self.running.retain(|r| r.seq != seq);
        }
        (emitted, finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive<F: FnMut(SeqId) -> u32>(s: &mut Scheduler, mut sample: F) -> (Vec<(u64, u32)>, Vec<Finished>) {
        let mut emitted = Vec::new();
        let mut finished = Vec::new();
        let mut steps = 0;
        while s.has_work() {
            steps += 1;
            assert!(steps < 1000, "runaway scheduler");
            let (plan, fin) = s.schedule();
            finished.extend(fin);
            let Some(plan) = plan else { continue };
            let sampled: Vec<(SeqId, u32)> =
                plan.samples.iter().map(|&(_r, seq)| (seq, sample(seq))).collect();
            let (em, fin) = s.ingest(&sampled);
            emitted.extend(em);
            finished.extend(fin);
        }
        (emitted, finished)
    }

    #[test]
    fn two_requests_complete_by_length() {
        let mut s = Scheduler::new(64, SchedulerConfig { max_batch: 4, max_prefill: 0 });
        s.add_request(Request { id: 1, prompt: vec![10, 11, 12], params: SamplingParams { max_tokens: 4, ignore_eos: false } });
        s.add_request(Request { id: 2, prompt: vec![20, 21], params: SamplingParams { max_tokens: 4, ignore_eos: false } });
        // Always sample a non-EOS token.
        let (emitted, finished) = drive(&mut s, |_| 50);
        assert_eq!(emitted.iter().filter(|(id, _)| *id == 1).count(), 4, "req1 emits 4 outputs");
        assert_eq!(emitted.iter().filter(|(id, _)| *id == 2).count(), 4, "req2 emits 4 outputs");
        assert!(finished.iter().all(|f| f.reason == FinishReason::Length));
        assert_eq!(finished.len(), 2);
        assert_eq!(s.kv_used(), 0, "all slots recycled");
    }

    #[test]
    fn eos_stops_after_first_output() {
        let mut s = Scheduler::new(64, SchedulerConfig { max_batch: 4, max_prefill: 0 });
        s.add_request(Request { id: 1, prompt: vec![10, 11, 12], params: SamplingParams { max_tokens: 50, ignore_eos: false } });
        // EOS is sampled every step; prefill intermediates are discarded, so the
        // first *real* output is EOS -> Stop after exactly one emitted token.
        let (emitted, finished) = drive(&mut s, |_| EOS_TOKENS[0]);
        assert_eq!(emitted.len(), 1);
        assert_eq!(finished, vec![Finished { id: 1, reason: FinishReason::Stop }]);
        assert_eq!(s.kv_used(), 0);
    }

    #[test]
    fn rejects_empty_prompt() {
        let mut s = Scheduler::new(8, SchedulerConfig { max_batch: 2, max_prefill: 0 });
        s.add_request(Request { id: 7, prompt: vec![], params: SamplingParams::default() });
        let (_plan, fin) = s.schedule();
        assert_eq!(fin, vec![Finished { id: 7, reason: FinishReason::Rejected }]);
    }
}
