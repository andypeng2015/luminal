//! Free-list paged-KV slot allocator for the serving engine.
//!
//! The paged KV cache is a flat `[num_slots, KV_DIM]` pool per layer (see
//! `model.rs`). One *slot* holds one token's KV across all layers. For
//! continuous batching we must allocate slots to sequences as they arrive and
//! return them to a free pool when sequences finish, so a long-running server
//! recycles capacity instead of leaking it (the demo's `PageTable` only ever
//! bumped a `next_free_slot` and never freed).

use rustc_hash::FxHashMap;

/// Opaque per-sequence identifier.
pub type SeqId = u64;

/// A free-list allocator over a fixed pool of `capacity` KV slots.
///
/// Each sequence owns an ordered list of slot indices (its context, oldest
/// first). `allocate` appends slots; `free` returns them all to the pool.
pub struct KvAllocator {
    capacity: usize,
    /// Stack of available slot indices.
    free: Vec<usize>,
    /// seq_id -> owned slot indices, in context (position) order.
    tables: FxHashMap<SeqId, Vec<usize>>,
}

impl KvAllocator {
    pub fn new(capacity: usize) -> Self {
        // Hand out low indices first (pop from the back -> ascending order).
        let free = (0..capacity).rev().collect();
        Self {
            capacity,
            free,
            tables: FxHashMap::default(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Slots not currently assigned to any sequence.
    pub fn free_slots(&self) -> usize {
        self.free.len()
    }

    pub fn used_slots(&self) -> usize {
        self.capacity - self.free.len()
    }

    /// Whether `n` more slots can be allocated right now.
    pub fn can_allocate(&self, n: usize) -> bool {
        self.free.len() >= n
    }

    /// Append `n` freshly-allocated slots to `seq`'s context. Returns the new
    /// slot indices in order, or `None` if the pool can't satisfy the request
    /// (in which case nothing is allocated).
    pub fn allocate(&mut self, seq: SeqId, n: usize) -> Option<Vec<usize>> {
        if self.free.len() < n {
            return None;
        }
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            slots.push(self.free.pop().expect("checked capacity above"));
        }
        self.tables.entry(seq).or_default().extend_from_slice(&slots);
        Some(slots)
    }

    /// Return all slots owned by `seq` to the free pool (sequence finished or
    /// evicted). No-op if the sequence is unknown.
    pub fn free(&mut self, seq: SeqId) {
        if let Some(slots) = self.tables.remove(&seq) {
            self.free.extend(slots);
        }
    }

    /// The slot indices backing `seq`'s context, oldest first.
    pub fn context_slots(&self, seq: SeqId) -> &[usize] {
        self.tables.get(&seq).map_or(&[], |v| v.as_slice())
    }

    /// Number of context tokens (slots) currently held by `seq`.
    pub fn context_len(&self, seq: SeqId) -> usize {
        self.tables.get(&seq).map_or(0, Vec::len)
    }

    /// Number of live sequences.
    pub fn live_sequences(&self) -> usize {
        self.tables.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_free_recycle() {
        let mut a = KvAllocator::new(8);
        assert_eq!(a.free_slots(), 8);
        assert_eq!(a.used_slots(), 0);

        // Sequence 1 takes 3 slots (ascending order).
        let s1 = a.allocate(1, 3).unwrap();
        assert_eq!(s1, vec![0, 1, 2]);
        assert_eq!(a.context_slots(1), &[0, 1, 2]);
        assert_eq!(a.context_len(1), 3);
        assert_eq!(a.used_slots(), 3);

        // Append 2 more to seq 1 (context grows in order).
        let s1b = a.allocate(1, 2).unwrap();
        assert_eq!(s1b, vec![3, 4]);
        assert_eq!(a.context_slots(1), &[0, 1, 2, 3, 4]);

        // Sequence 2 takes the remaining 3.
        assert!(a.can_allocate(3));
        let s2 = a.allocate(2, 3).unwrap();
        assert_eq!(s2, vec![5, 6, 7]);
        assert_eq!(a.free_slots(), 0);

        // Pool exhausted.
        assert!(!a.can_allocate(1));
        assert!(a.allocate(3, 1).is_none());

        // Free seq 1 -> its 5 slots return to the pool; seq 2 untouched.
        a.free(1);
        assert_eq!(a.free_slots(), 5);
        assert_eq!(a.context_len(1), 0);
        assert_eq!(a.context_slots(2), &[5, 6, 7]);
        assert_eq!(a.live_sequences(), 1);

        // Reuse recycled capacity.
        let s3 = a.allocate(3, 4).unwrap();
        assert_eq!(s3.len(), 4);
        assert_eq!(a.free_slots(), 1);
    }

    #[test]
    fn allocate_all_or_nothing() {
        let mut a = KvAllocator::new(4);
        assert!(a.allocate(1, 5).is_none());
        assert_eq!(a.free_slots(), 4, "failed allocation must not consume slots");
        assert!(a.allocate(1, 4).is_some());
    }

    #[test]
    fn free_unknown_seq_is_noop() {
        let mut a = KvAllocator::new(4);
        a.free(999);
        assert_eq!(a.free_slots(), 4);
    }
}
