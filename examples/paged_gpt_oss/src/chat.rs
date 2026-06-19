//! gpt-oss harmony chat templating.

/// Wrap a single user turn in the minimal harmony chat format the model expects
/// (matches the demo and the HF golden reference).
pub fn harmony_prompt(user_prompt: &str) -> String {
    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
         Reasoning: low<|end|>\
         <|start|>user<|message|>{user_prompt}<|end|>\
         <|start|>assistant"
    )
}
