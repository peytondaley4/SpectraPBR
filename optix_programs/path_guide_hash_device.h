#pragma once

//------------------------------------------------------------------------------
// Device-resident path-guide cell table (hash + bump allocator)
//
// Cells come into existence the moment a shader first needs them: the lookup
// miss path inserts the key with atomicCAS and allocates a data slot from an
// atomic counter. There is no staging buffer, no host readback, no CPU merge,
// and no double-buffered swap — cell indices are stable for the lifetime of
// the grid, and a freshly inserted cell is usable by the next refit.
//
// Shared by raygen.cu (OptiX) and path_guide_kernels.cu (plain CUDA): every
// function takes explicit pointers/arguments — no launch-params dependency,
// no shared memory, no warp-level intrinsics (OptiX-safe).
//
// Concurrency notes:
//  - keys[] transitions EMPTY -> key exactly once (atomicCAS); values[]
//    transitions PENDING -> index (or FULL) exactly once, published with
//    atomicExch after a __threadfence so the winner's payload initialization
//    (lobe init + cell_keys) is visible before the index is.
//  - Lookups use plain loads. A reader can transiently see a stale EMPTY or
//    PENDING for a cell that another SM just inserted; both read as "miss",
//    which is benign — the vertex falls back to BSDF sampling for one frame.
//    Readers never spin.
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include "path_guide_cell_layout.h"

#define PG_KEY_EMPTY       0xFFFFFFFFFFFFFFFFull
#define PG_VALUE_PENDING   0xFFFFFFFFu
#define PG_VALUE_FULL      0xFFFFFFFEu
#define PG_VALUE_MIN_SENTINEL 0xFFFFFFFEu   // values >= this are not cell indices
#define PG_MAX_PROBES      32u
#define PG_INVALID_CELL    0xFFFFFFFFu

struct PathGuideTableDevice {
    unsigned long long* hash_keys;   // [hash_table_size], (level<<48 | morton), PG_KEY_EMPTY when free
    unsigned int* hash_values;       // [hash_table_size], cell index or sentinel
    unsigned int hash_table_size;    // power of two, >= 2x cell_capacity
    unsigned int hash_shift;         // 64 - log2(hash_table_size)
    unsigned long long* cell_keys;   // [cell_capacity], packed key per allocated cell (winner-written)
    unsigned int* cell_counter;      // bump allocator (1 element)
    unsigned int cell_capacity;
    float* data;                     // [cell_capacity * entry_stride]
    unsigned int entry_stride;
};

// ─── Morton encode (64-bit, up to 21 bits/axis). Must match the host. ───────
__forceinline__ __device__ unsigned long long pgMortonSpread3(unsigned long long x) {
    x &= 0x1fffffull;
    x = (x | x << 32) & 0x001f00000000ffffull;
    x = (x | x << 16) & 0x001f0000ff0000ffull;
    x = (x | x << 8)  & 0x010f00f00f00f00full;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ull;
    x = (x | x << 2)  & 0x1249249249249249ull;
    return x;
}

__forceinline__ __device__ unsigned long long pgMortonEncode64(
    unsigned int ix, unsigned int iy, unsigned int iz)
{
    return pgMortonSpread3(ix) | (pgMortonSpread3(iy) << 1) | (pgMortonSpread3(iz) << 2);
}

__forceinline__ __device__ unsigned long long pgPackKey(
    unsigned int level, unsigned long long morton)
{
    return ((unsigned long long)level << 48) | morton;
}

// Morton interleaves one bit per axis, so descending one level is a 3-bit
// shift: child = (parent << 3) | octant, octant = dx | dy<<1 | dz<<2.
__forceinline__ __device__ unsigned long long pgChildMorton(
    unsigned long long parentMorton, unsigned int octant)
{
    return (parentMorton << 3) | (unsigned long long)octant;
}

__forceinline__ __device__ unsigned int pgHashSlot(
    unsigned long long key, unsigned int shift, unsigned int mask)
{
    // Fibonacci hashing: multiply by the 64-bit golden ratio, take top bits
    return (unsigned int)((key * 0x9E3779B97F4A7C15ull) >> shift) & mask;
}

// Lookup: returns the cell index or PG_INVALID_CELL. Plain loads (see header
// comment for why transient staleness is acceptable).
__forceinline__ __device__ unsigned int pgTableLookup(
    const PathGuideTableDevice& t,
    unsigned int level, unsigned long long morton)
{
    if (t.hash_keys == nullptr || t.hash_table_size == 0) return PG_INVALID_CELL;

    unsigned long long key = pgPackKey(level, morton);
    unsigned int mask = t.hash_table_size - 1;
    unsigned int slot = pgHashSlot(key, t.hash_shift, mask);

    for (unsigned int i = 0; i < PG_MAX_PROBES; i++) {
        unsigned long long k = t.hash_keys[slot];
        if (k == key) {
            unsigned int v = t.hash_values[slot];
            return (v >= PG_VALUE_MIN_SENTINEL) ? PG_INVALID_CELL : v;
        }
        if (k == PG_KEY_EMPTY) return PG_INVALID_CELL;
        slot = (slot + 1) & mask;
    }
    return PG_INVALID_CELL;
}

// Insert-or-lookup: returns the cell index, or PG_INVALID_CELL when the cell
// is mid-insertion by another thread (PENDING), the allocator/table is full,
// or the probe cap is hit. The winner initializes the lobe parameters
// (pgInitCellLobes) BEFORE publishing the index, so no thread can ever
// hard-assign a deposit against all-zero lobes; the rest of the payload is
// pre-zeroed by the host. *wasInserted is set when THIS thread won the
// insert — callers that overwrite the default lobes (subdivision warm
// starts) do so when it is true; those fields are disjoint from the interval
// sums other threads may concurrently atomicAdd into.
__forceinline__ __device__ unsigned int pgTableInsert(
    const PathGuideTableDevice& t,
    unsigned int level, unsigned long long morton,
    bool* wasInserted)
{
    if (wasInserted) *wasInserted = false;
    if (t.hash_keys == nullptr || t.hash_table_size == 0 || t.cell_counter == nullptr)
        return PG_INVALID_CELL;
    // Cheap early-out once the allocator is exhausted (avoids CAS traffic;
    // the post-allocation check below still handles the race window).
    if (*t.cell_counter >= t.cell_capacity) return PG_INVALID_CELL;

    unsigned long long key = pgPackKey(level, morton);
    unsigned int mask = t.hash_table_size - 1;
    unsigned int slot = pgHashSlot(key, t.hash_shift, mask);

    for (unsigned int i = 0; i < PG_MAX_PROBES; i++) {
        unsigned long long prev = atomicCAS(&t.hash_keys[slot], PG_KEY_EMPTY, key);
        if (prev == PG_KEY_EMPTY) {
            // Won the slot — allocate, initialize, then publish.
            unsigned int idx = atomicAdd(t.cell_counter, 1u);
            if (idx >= t.cell_capacity) {
                atomicExch(&t.hash_values[slot], PG_VALUE_FULL);
                return PG_INVALID_CELL;
            }
            pgInitCellLobes(t.data + (unsigned long long)idx * t.entry_stride);
            t.cell_keys[idx] = key;
            __threadfence();
            atomicExch(&t.hash_values[slot], idx);
            if (wasInserted) *wasInserted = true;
            return idx;
        }
        if (prev == key) {
            unsigned int v = t.hash_values[slot];
            return (v >= PG_VALUE_MIN_SENTINEL) ? PG_INVALID_CELL : v;
        }
        slot = (slot + 1) & mask;
    }
    return PG_INVALID_CELL;
}
