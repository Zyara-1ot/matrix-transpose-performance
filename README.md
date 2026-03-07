# Matrix Transpose Optimization  From 0.16 GB/s to 11.5 GB/s

A deep dive into optimizing a 4096×4096 integer matrix transpose on an AMD Ryzen 5 5500U (Fedora Linux), achieving **71x speedup** by systematically eliminating every memory bottleneck.

**Final result: 11.5 GB/s .**

---

## Hardware

| Property | Value |
|---|---|
| CPU | AMD Ryzen 5 5500U |
| Cores / Threads | 6 / 12 |
| Max Clock | 4056 MHz |
| SIMD | AVX2 (256-bit) |
| L1 Cache | 32KB per core |
| L2 Cache | 512KB per core |
| L3 Cache | 8MB shared |
| Memory | DDR4 dual channel |
| OS | Fedora Linux |

---

## Build

```bash
g++ -O3 -march=native -fopenmp transpose.cpp -o transpose
sudo sh -c 'echo 128 > /proc/sys/vm/nr_hugepages'  # for huge pages
./transpose
```

---

## Results  Every Optimization Step

| Step | Optimization | GB/s | Speedup vs Naive |
|---|---|---|---|
| 1 | Naive double loop | 0.16 | 1x |
| 2 | Cache tiling (T=64) | 0.82 | 5x |
| 3 | OpenMP + -O3 + -march=native | 2.29 | 14x |
| 4 | Input padding (pad_m = m+16) | 3.0 | 19x |
| 5 | AVX2 8×8 register transpose | 7.23 | 45x |
| 6 | Output padding (pad_n = n+16) | 7.43 | 46x |
| 7 | Remove outer tiling (2 loops) | ~10.0 | 62x |
| 8 | 2MB Huge pages | ~11.5 | 71x |

**Hardware memcpy ceiling (64MB, OpenMP):** 8.048 GB/s  
**Final transpose:** 11.5 GB/s

---

## Why Each Optimization Works

---

### Step 1 - Naive (0.16 GB/s)

```cpp
for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++)
        B[j * n + i] = A[i * m + j];
```

Reading A row by row is sequential  good. Writing to B column by column jumps `n * sizeof(int) = 16KB` per write  every single write is a cache miss, forcing a 400-cycle RAM stall. CPU spends 99% of time waiting for memory.

---

### Step 2 - Cache Tiling (0.82 GB/s)

```cpp
for(int I = 0; I < n; I += T)
    for(int J = 0; J < m; J += T)
        for(int i = I; i < I+T; i++)
            for(int j = J; j < J+T; j++)
                B[j*n+i] = A[i*m+j];
```

Process T×T blocks instead of full rows. A 64×64 tile = 64×64×4 = 16KB - fits in L1 cache (32KB) alongside the destination tile. Both read and write data stay warm in cache during the entire tile. Jumps are contained within the tile, not across 4096 rows.

Time complexity is still O(N²)  same work, smarter order.

---

### Step 3 - OpenMP + Compiler Flags (2.29 GB/s)

```cpp
#pragma omp parallel for collapse(2)
for(int I ...) for(int J ...)
```

- **OpenMP:** 5 cores were sitting idle. `collapse(2)` parallelizes the two outer tile loops, distributing independent tiles across all 6 cores. No data races — each tile writes to a unique region of B.
- **-O3:** Auto-vectorization, instruction reordering, loop unrolling, automatic prefetch hints.
- **-march=native:** Compiler targets your exact CPU, enabling AVX2, FMA, and all Ryzen-specific instructions.

---

### Step 4 - Input Padding (3.0 GB/s)

Matrix rows are 4096 elements = 16384 bytes wide. 16384 is a power of 2. L1 cache has 64 sets. Cache set formula:

```
set = (address / 64) % 64
```

Row 0: set 0. Row 1: address += 16384, set = (16384/64) % 64 = 256 % 64 = **0**. Row 2: **0**. Every row maps to set 0! 63 sets sit empty while all rows fight over one set, constantly evicting each other , cache thrashing.

Fix: pad row width to `m + 16 = 4112` elements. Row stride becomes 16448 bytes. Now row 0 → set 0, row 1 → set 1, row 2 → set 2. All 64 sets used, no conflicts.

The 16 padding ints (64 bytes = one cache line) are never read or written for actual data.

---

### Step 5 - AVX2 8×8 Register Transpose (7.23 GB/s)

Instead of processing one element at a time, load 8 full rows into 8 YMM registers , transpose entirely inside registers using shuffle instructions, then store 8 columns.

```
Memory operations: 8 loads + 8 stores = 16 total for 64 elements
vs scalar: 64 reads + 64 writes = 128 total
```

8x reduction in memory operations. The shuffle happens inside the CPU register-to-register, 1-3 cycles each, zero memory traffic.

**Three-step shuffle sequence:**

```
Step 1 — unpacklo/hi_epi32:  interleave at int level (2×2 blocks)
Step 2 — unpacklo/hi_epi64:  interleave at 64-bit level (4×4 blocks)
Step 3 — permute2x128:       fix lane crossing (complete 8×8)
```

AVX2 registers split into two 128-bit lanes. Steps 1 and 2 work within each lane. `permute2x128` is the only instruction that moves data between lanes , essential for completing the transpose.

---

### Step 6 - Output Padding (7.43 GB/s)

Same power-of-2 problem on B. B's row width is `n = 4096`  same cache set conflicts on writes. Even though we never explicitly read B, every write causes "read for ownership" — CPU fetches the cache line from RAM before modifying it. Those fetches suffer the same set conflicts.

Fix: `pad_n = n + 16 = 4112` as B's stride. Same logic as pad_m.

---

### Step 7 - Remove Outer Tiling (10.0 GB/s)

Replaced 4 nested loops with 2:

```cpp
// Before: 4 loops (64x64 tile wrapper + 8x8 AVX inner)
// After: 2 loops (just 8x8 AVX directly)
#pragma omp parallel for collapse(2)
for(int i = 0; i < n; i += 8)
    for(int j = 0; j < m; j += 8)
        avxtps(A + i*pad_m + j, B + j*pad_n + i, pad_m, pad_n);
```

AVX already provides natural 8×8 cache-friendly blocking. The outer 64×64 tiling was redundant , adding loop overhead and constraining OpenMP's work distribution.

With 2 loops, OpenMP has 512×512 = **262,144 independent tasks** to distribute vs 64×64 = 4,096 before. Finer granularity = better load balancing across cores = better throughput.

This also explains exceeding the memcpy benchmark the memcpy used unaligned vectors, our code uses 64-byte aligned + padded memory with perfectly distributed parallel work.

---

### Step 8 - 2MB Huge Pages (11.5 GB/s)

Every memory access requires translating virtual address → physical address via the page table. The TLB (Translation Lookaside Buffer) caches these translations  but only holds ~64-128 entries.

- **4KB normal pages:** 64MB matrix needs 16,384 page table entries  constant TLB misses, each requiring a RAM access to walk the page table (~100ns penalty)
- **2MB huge pages:** Same matrix needs only **32 entries**  fits entirely in TLB, zero TLB misses

```bash
sudo sh -c 'echo 128 > /proc/sys/vm/nr_hugepages'
# allocate with mmap + MAP_HUGETLB
```

This was the final remaining bottleneck  once tiling, padding, and AVX eliminated cache and compute bottlenecks, TLB pressure became visible.

---

## What Didn't Work (and Why)

### Streaming Stores

`_mm256_stream_si256` bypasses cache  writes directly to RAM, skipping read-for-ownership. Should be perfect for write-only B.

**Why it failed:** Our write pattern `B[j*pad_n+i]` jumps 16KB between consecutive avxtps calls. Cache batches these scattered writes efficiently - multiple writes collect in the same cache line before one RAM write. Streaming stores remove that batching , every write hits RAM individually. With 6 OpenMP threads all doing scattered streaming stores simultaneously, the memory controller gets overwhelmed. Result: 6.2 GB/s vs 7.4 GB/s with regular stores.

Streaming stores work for sequential writes (memcpy-style), not scattered writes (transpose-style).

---

### Manual Prefetching

`__builtin_prefetch` hints to fetch next data from RAM while processing current data.

**Why it failed:** `-O3 -march=native` already inserts automatic prefetch instructions based on loop pattern analysis. Adding manual prefetches created double-prefetching  two fetch requests for the same data, competing for memory bandwidth and cache space. Result: 7-8 GB/s vs 10 GB/s without.

Manual prefetch helps for irregular access patterns (pointer chasing, graphs) where the compiler can't predict what you need next. For regular strided loops like ours, the compiler does it better.

---

### Huge Pages (First Attempt)

Failed when code still had outer 64×64 tiling. TLB benefit was outweighed by other inefficiencies. Removing tiling first made TLB the visible bottleneck  then huge pages could fix it.



## Notes

- Implementation assumes matrix dimensions divisible by 8 (4096 qualifies). Scalar fallback for arbitrary sizes , will add edge cases soon.
- Huge pages require: `sudo sh -c 'echo 128 > /proc/sys/vm/nr_hugepages'`
- Benchmark: 100-run average, measures transpose call only
- All results verified correct: `B[j*pad_n+i] == A[i*pad_m+j]` for all i,j

---

The bottleneck is always the same: memory is slow, compute is fast. The solution is always the same: keep data close to the processor, maximize reuse, minimize RAM trips.
