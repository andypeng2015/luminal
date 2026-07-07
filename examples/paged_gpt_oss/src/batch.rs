//! Host-side batch assembly for the paged model. Builds the scatter/gather slot
//! indices, query positions, and the per-request `qo_indptr`/`kv_indptr`
//! boundary vectors (from which the model builds its causal / sliding-window
//! mask in-graph) for a mixed continuous-batching step. Adapted from the demo's
//! `build_batch` to read slots from [`KvAllocator`].

use crate::kv_alloc::{KvAllocator, SeqId};
use crate::model::VOCAB_SIZE;

pub struct Batch {
    pub scatter_idx: Vec<i32>,
    pub gather_idx: Vec<i32>,
    pub q_pos: Vec<i32>,
    /// `[0, cum. query counts per entry]` — request boundaries in query rows.
    pub qo_indptr: Vec<i32>,
    /// `[0, cum. context lengths per entry]` — request boundaries in ctx rows.
    pub kv_indptr: Vec<i32>,
    pub total_s: usize,
    pub total_c: usize,
}

/// `entries`: `(seq, query positions)` in batch-row order. A sequence's slots
/// are allocated in position order, so a context slot's index within its
/// sequence equals that token's absolute position.
pub fn build_batch(entries: &[(SeqId, Vec<usize>)], alloc: &KvAllocator) -> Batch {
    let total_s: usize = entries.iter().map(|(_, pos)| pos.len()).sum();

    // Request boundaries: kv_indptr over gathered context rows, qo_indptr over
    // query rows — the model derives the attention mask from these in-graph.
    let mut gather_idx: Vec<i32> = vec![];
    let mut kv_indptr: Vec<i32> = vec![0];
    for (seq, _) in entries {
        let slots = alloc.context_slots(*seq);
        gather_idx.extend(slots.iter().map(|&s| s as i32));
        kv_indptr.push(gather_idx.len() as i32);
    }
    let total_c = gather_idx.len();

    let mut scatter_idx: Vec<i32> = vec![];
    let mut q_pos: Vec<i32> = vec![];
    let mut qo_indptr: Vec<i32> = vec![0];
    for (seq, positions) in entries {
        let ctx_len = alloc.context_len(*seq);
        let n_new = positions.len();
        let slots = alloc.context_slots(*seq);
        scatter_idx.extend(slots[ctx_len - n_new..].iter().map(|&s| s as i32));
        q_pos.extend(positions.iter().map(|&p| p as i32));
        qo_indptr.push(q_pos.len() as i32);
    }

    Batch {
        scatter_idx,
        gather_idx,
        q_pos,
        qo_indptr,
        kv_indptr,
        total_s,
        total_c,
    }
}

pub fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32
}

pub fn logits_row(all_logits: &[f32], row_idx: usize) -> &[f32] {
    &all_logits[row_idx * VOCAB_SIZE..(row_idx + 1) * VOCAB_SIZE]
}
