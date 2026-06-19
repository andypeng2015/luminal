//! Host-side batch assembly for the paged model. Builds the scatter/gather slot
//! indices, query positions, and the two additive attention masks (full causal,
//! and causal + sliding-window) for a mixed continuous-batching step. Adapted
//! from the demo's `build_batch` to read slots from [`KvAllocator`].

use crate::kv_alloc::{KvAllocator, SeqId};
use crate::model::{SLIDING_WINDOW, VOCAB_SIZE};

pub struct Batch {
    pub scatter_idx: Vec<i32>,
    pub gather_idx: Vec<i32>,
    pub q_pos: Vec<i32>,
    pub mask_full: Vec<f32>,
    pub mask_sliding: Vec<f32>,
    pub total_s: usize,
    pub total_c: usize,
}

/// `entries`: `(seq, query positions)` in batch-row order. A sequence's slots
/// are allocated in position order, so a context slot's index within its
/// sequence equals that token's absolute position.
pub fn build_batch(entries: &[(SeqId, Vec<usize>)], alloc: &KvAllocator) -> Batch {
    let total_s: usize = entries.iter().map(|(_, pos)| pos.len()).sum();

    let mut gather_idx: Vec<i32> = vec![];
    let mut ctx_ranges: Vec<(usize, usize)> = vec![];
    for (seq, _) in entries {
        let start = gather_idx.len();
        let slots = alloc.context_slots(*seq);
        gather_idx.extend(slots.iter().map(|&s| s as i32));
        ctx_ranges.push((start, slots.len()));
    }
    let total_c = gather_idx.len();

    let mut scatter_idx: Vec<i32> = vec![];
    let mut q_pos: Vec<i32> = vec![];
    for (seq, positions) in entries {
        let ctx_len = alloc.context_len(*seq);
        let n_new = positions.len();
        let slots = alloc.context_slots(*seq);
        scatter_idx.extend(slots[ctx_len - n_new..].iter().map(|&s| s as i32));
        q_pos.extend(positions.iter().map(|&p| p as i32));
    }

    // Masks default to -1e30 (blocked); a query attends only within its own
    // sequence's context range (cross-sequence isolation), causally, and — for
    // the sliding mask — within the window.
    let mut mask_full = vec![-1e30f32; total_s * total_c];
    let mut mask_sliding = vec![-1e30f32; total_s * total_c];
    let mut q_offset = 0;
    for (entry_idx, (_, positions)) in entries.iter().enumerate() {
        let (ctx_start, ctx_len) = ctx_ranges[entry_idx];
        for (qi, &abs_pos) in positions.iter().enumerate() {
            for ci in 0..ctx_len {
                if ci <= abs_pos {
                    let idx = (q_offset + qi) * total_c + (ctx_start + ci);
                    mask_full[idx] = 0.0;
                    if abs_pos - ci < SLIDING_WINDOW {
                        mask_sliding[idx] = 0.0;
                    }
                }
            }
        }
        q_offset += positions.len();
    }

    Batch {
        scatter_idx,
        gather_idx,
        q_pos,
        mask_full,
        mask_sliding,
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
