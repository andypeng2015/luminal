//! Library surface for the paged gpt-oss serving engine. The `main.rs` demo and
//! the `serve` binary share these modules.

pub mod hf;
pub mod model;
pub mod quant;

pub mod batch;
pub mod chat;
pub mod engine;
pub mod kv_alloc;
pub mod scheduler;
pub mod server;
