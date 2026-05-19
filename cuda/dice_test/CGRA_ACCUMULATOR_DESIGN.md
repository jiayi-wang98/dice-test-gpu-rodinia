# CGRA Accumulator-PE Design — Intra-CTA Atomics, Scan, and Reduction

**Status:** Design proposal (2026-05-18). Pre-implementation.

**Motivation:** Replace GPU's L2-atomic-bound block-level reduce/scan/atomic pattern with a single stateful PE configuration in DICE's CGRA, eliminating SMEM round-trips and most L2 atomic traffic.

---

## 1. Motivation and value proposition

### The GPU baseline pattern

For any CTA-local atomic/reduce/scan operation on a Turing-class GPU, the canonical pattern is one of:

| Strategy | L2 atomics per CTA | Block-local cost |
|---|---:|---|
| Naïve thread-level atomic | `blockDim.x` | none |
| Warp-shuffle + atomicAdd | `blockDim.x / 32` | log₂(32) = 5 shuffles per warp |
| SMEM block-reduce tree + atomicAdd | 1 | log₂(blockDim.x) = 10 barriers + bank-conflict-aware SMEM reads + warp shuffles + tree-reduce |
| **DICE accumulator PE** | **1** | **0** — single PE in self-feedback mode |

The block-reduce-tree path matches the L2 atomic traffic of the DICE PE, but pays a steep block-local cost (SMEM traffic, log barriers, warp shuffles, register pressure). DICE collapses the block-local cost to ~zero.

### Where the savings live

- **L2 atomic traffic**: Same as the best GPU pattern (1 per CTA) — no win vs the SMEM-tree baseline, but a big win vs warp-shuffle (32×) and naïve (1024×).
- **SM-dynamic energy**: Big win — no SMEM bank cycles, no barriers, no warp shuffles, no tree-reduction RF traffic.
- **Programming model**: Big win — replaces 30-line tree-reduction boilerplate with one intrinsic call.

For a 1024-thread CTA reducing N=1M elements, the L2 dynamic energy is reduced by ~1000× versus naïve, matching warp-shuffle's L2 count but **at the SM-dynamic energy of a single-PE pipeline op per thread**.

### The key insight from DICE's execution model

> **DICE dispatches threads of a CTA through the CGRA in deterministic order, one per pipe cycle. So `atomicAdd` on a CTA-local target collapses to "stateful PE in pipelined dispatch" — no race, no lock, no retry, by construction.**

GPU atomics are complex *because* threads race. DICE's pipelined dispatch already serializes within a CTA → atomicity is **free**, not a primitive that requires hardware support.

---

## 2. Architectural model — PE self-feedback via crossbar

### No new hardware structure

The "accumulator" is **not** a new register file. It's a new **PE configuration** that uses entirely existing CGRA primitives:

| Component | Source |
|---|---|
| Accumulator state | PE's existing **output register** |
| Self-feedback wire | **CGRA crossbar** routed back to PE input |
| ALU op (add/min/max/and/or/xor/exch/cas) | PE's existing **ALU op selector** |
| Per-thread writeback to RF | DICE's existing **per-thread predicate + RF write port** |
| Init value | One-cycle init from compile-time constant or RF source at DBB entry |

Net new hardware: **~0**. Cost is a few bits of bitstream encoding to select self-feedback routing + accumulator-mode ALU.

### Datapath diagram

```
                                 ┌─────────────────────────────┐
                                 │       PE (accumulator)      │
                                 │                             │
       thread.val ───►───────────│──►─┐                        │
                                 │    │   ┌──ALU──┐            │
   ┌──── CGRA crossbar ──────────│────┴──►│ +     │──┐         │
   │                             │        │ min   │  │         │
   │                             │        │ max …│  │         │
   │                             │        └───┬───┘  │         │
   │                             │            │      │         │
   │                             │       ┌────▼───┐  │         │
   │                             │       │  REG   │◄─┘         │   ← state register
   │                             │       │ (state)│            │
   │                             │       └────┬───┘            │
   │                             │            │                │
   ◄────────────────────[crossbar wires]──────┘ (self-feedback)│
                                 │            │                │
                                 │     (predicate gates write) │
                                 │            ▼                │
                                 │     writeback to RF         │
                                 └─────────────────────────────┘
```

### Per-cycle operation

For each thread dispatched through the PE in accumulator mode:

1. Read state register (combinational output of PE register)
2. Read `val` input (from dispatching thread)
3. ALU compute: `new_state = ALU(state, val)`
4. Register update (clocked): `state := new_state` (visible to next thread)
5. **Writeback to RF (predicate-gated)**: emit either NEW or OLD state to the dispatching thread's RF, depending on op
6. Crossbar feedback wires PE output back to PE input for the next cycle

### Writeback semantics by op

| Op | What writes back to RF | Rationale |
|---|---|---|
| `add`/`sub`/`min`/`max`/`and`/`or`/`xor` | **NEW state** | Inclusive scan; last thread holds the total |
| `exch`/`cas` | **OLD state** | Caller needs to know what was overwritten (CAS success detection) |

The bitstream encoding selects which tap (post-ALU output for NEW, pre-ALU register value for OLD).

### State lifetime: one DBB

In v1, accumulator state lives for **one DBB** — the duration of a single CGRA bitstream configuration. The CTA's full stream of threads passes through the accumulator DBB; at end of DBB, the PE register holds the final state.

For multi-DBB accumulator patterns (e.g., long-running counters across phases of a kernel), the compiler either:

- (a) Keeps the PE in accumulator mode across multiple DBBs (CGRA placer constraint — possible but adds complexity)
- (b) Reads out the state at end of each DBB via a normal PE output to RF, re-inits the accumulator in the next DBB from that RF location

v1 ships option (b) — simpler placer; the multi-DBB optimization is left for future work.

---

## 3. Programming model

### Source-level intrinsic API

```c
// Declare an accumulator: a local variable whose initial value is the PE state at DBB entry.
T X = init_val;

// ALU ops — all forms have the same shape: X = post-op PE state (NEW); val combines into PE state.
dice_cta_acc_add  (X, val);
dice_cta_acc_sub  (X, val);
dice_cta_acc_min  (X, val);
dice_cta_acc_max  (X, val);
dice_cta_acc_and  (X, val);
dice_cta_acc_or   (X, val);
dice_cta_acc_xor  (X, val);

// Unconditional swap — X becomes OLD PE state.
dice_cta_acc_exch (X, val);

// CAS — X becomes OLD PE state; PE state updates iff state == cmp.
dice_cta_acc_cas  (X, cmp, val);
```

### Variable name = accumulator identity

The first argument is a thread-local variable that simultaneously serves three roles:

1. **At declaration** (`T X = init_val;`) — declares a logical accumulator with the given init
2. **Inside the intrinsic** — names which accumulator to fire
3. **After the call** — overwritten per thread with the PE state (NEW or OLD per op)

Each unique variable used as the first argument across the kernel binds to one PE in accumulator mode. The compiler does the placement.

### Predication

The writeback to RF is gated by the thread's standard DICE predicate. The PE state still updates even when the writeback predicate is false — this matters because the accumulator is intrinsically about **all** threads contributing, not just the predicated-true ones.

> **Open design question:** should masking apply only to RF writeback, or also to PE state update? For "atomicAdd-style" semantics, predicate-false threads should NOT contribute to the state. For "scan-style" semantics, all threads do contribute. Default proposal: **predicate gates both** — if thread is masked off, the accumulator skips its contribution (matches CUDA atomicAdd semantics when called under control flow).

### Three canonical patterns

#### A. Inclusive prefix scan (every thread reads its position)

```c
__global__ void scan(const int *in, int *out, int N) {
    int tid = threadIdx.x;
    int v   = in[tid];

    int prefix = 0;
    dice_cta_acc_add(prefix, v);          // prefix = inclusive sum at this thread's position
    out[tid] = prefix;
}
```

Compiles to one DBB with one PE in accumulator-add mode. All threads streamed through; each thread's `prefix` is its inclusive scan value.

#### B. Block-level reduction (last thread has the total)

```c
__global__ void block_sum(const float *in, float *out, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float v = (tid < N) ? in[tid] : 0;

    int total = 0;
    dice_cta_acc_add(total, v);            // per-thread inclusive
    __syncthreads();

    if (threadIdx.x == blockDim.x - 1)
        atomicAdd(out, total);             // last thread's total = the full block sum
}
```

One CGRA-PE for the accumulator. One global L2 atomic per CTA. No SMEM, no warp shuffles, no tree.

#### C. Compare-and-swap (frontier-marking style)

```c
__global__ void claim_slot(int *result, int my_id) {
    int slot = -1;
    int prev = dice_cta_acc_cas(slot, -1, my_id);  // try to claim
    if (prev == -1) {
        // This thread successfully claimed the slot
        atomicAdd(result, 1);
    }
}
```

PE state initialized to −1; each thread's CAS sees the current state; only the first arriving thread (by pipeline order) succeeds. No retry loop needed.

### Compile-time vs runtime init

```c
int X = 0;                          // compile-time constant — PE state set at DBB entry, no instructions
int X = some_runtime_value;         // runtime — one-cycle init at DBB entry: thread 0's RF → PE state
```

---

## 4. Lowering chain (CUDA → PTX → PPTX → simulator)

### Option chosen: magic-address Trojan horse (v1)

For build-flow compatibility — NVCC + ptxas accept everything; DICE-side recognition happens at the PTX→PPTX step.

```c
// Header: dice_atomics.h
extern __shared__ unsigned int __dice_acc_slots[];     // reserved magic SMEM region

__device__ __forceinline__ unsigned int
dice_cta_acc_add(unsigned int &X, unsigned int val) {
    // PTX-level: atom.shared.add.u32 ... on the magic address.
    unsigned int slot_addr = (unsigned int)(__dice_acc_slots + /*X's slot index*/);
    return atomicAdd((unsigned int *)slot_addr, val);
}
// (Concrete impl uses templates or macros so the slot index comes from the variable's binding.)
```

The DICE p-graph compiler:

1. Scans PTX for `atom.shared.{op}` instructions
2. Resolves the address operand — if it falls in `__dice_acc_slots`, the compiler rewrites it as `acc.{op}` in the emitted PPTX
3. Allocates a PE in accumulator mode in the bitstream, with ALU op = the original atomic op
4. Updates the DBB metadata to declare the accumulator PE

For any other `atom.shared.*` (regular SMEM atomic), keep as-is — falls through to the existing DICE SMEM atomic path.

### Option deferred: custom PTX opcode (v2)

When richer accumulator features need expression beyond what `atom.shared.*` can mimic, switch to:

```c
asm volatile("dice.acc.add.u32 %0, %1;" : "=r"(X) : "r"(val));
```

This requires either compiling with PTX-only output (no SASS, no `ptxas` validation) or patching the simulator's PTX preprocessor to accept `dice.*` opcodes.

### PPTX-level instruction class

```
acc.init   %pe<i>, immediate              ; state[PE i] := imm  (executed once at DBB entry)
acc.add    %dst, %pe<i>, %src             ; %dst := new_state; state[PE i] += %src
acc.min    %dst, %pe<i>, %src             ;   …
acc.max    %dst, %pe<i>, %src             ;   …
acc.and / .or / .xor                      ;   …
acc.exch   %dst, %pe<i>, %src             ; %dst := OLD state; state[PE i] := %src
acc.cas    %dst, %pe<i>, %cmp, %val       ; %dst := OLD state; if (state==cmp) state := val
```

`%pe<i>` names the PE allocated to that accumulator role in the DBB's bitstream config.

### .meta declaration

```
DBB_ID = K,
ACCUMULATOR_PES = ((pe_3, add, init=0), (pe_5, min, init=INT_MAX), ...);
```

Each PE in accumulator mode gets one line: PE id, ALU op, init value. Lifetime within the DBB.

---

## 5. Simulator implementation (`dice_gpgpu-sim`)

### New PE configuration mode

`src/gpgpu-sim/cgra_core.cc` — add a per-PE config field:

```c++
enum pe_mode_t {
    PE_NORMAL,
    PE_ACCUMULATOR,        // self-feedback via crossbar
};

struct pe_config_t {
    pe_mode_t mode;
    alu_op_t  alu_op;        // existing
    uint32_t  acc_init;      // accumulator initial state (if mode == PE_ACCUMULATOR)
    bool      acc_writeback_new;  // true = emit NEW state, false = emit OLD (for exch/cas)
};
```

### PE state register

Already exists as the PE's output flop. In accumulator mode, the wraparound is enabled (crossbar routes output back to input).

### Per-cycle behavior

In `cgra_core_t::execute_pe_op(...)`:

```c++
if (pe.mode == PE_ACCUMULATOR) {
    uint32_t state = pe.state_reg;
    uint32_t new_state = alu(pe.alu_op, state, val_input);
    pe.state_reg = new_state;
    if (pe.acc_writeback_new) {
        emit_rf_writeback(thread_id, dst_reg, new_state, predicate);
    } else {
        emit_rf_writeback(thread_id, dst_reg, state, predicate);
    }
}
```

### CTA-scope reset

At CTA dispatch boundary, the CGRA scheduler clears accumulator state for all PEs in accumulator mode and applies their declared init values.

### DICEwattch counter

Add a new perf counter `DICE_ACC_OPS_N` for total accumulator-PE op count. Used to:

- Charge accumulator-PE energy (same per-op as a regular PE add — basically free)
- Quantify L2 atomic traffic reduction in benchmarks (compare against baseline run's `MEM_RD/MEM_WR/NOC_A` deltas)

---

## 6. Scope decisions for v1

### In scope

- Single-target intra-CTA accumulator with 8 ALU ops (`add/sub/min/max/and/or/xor/exch/cas`)
- Up to N independent accumulators per kernel (one per PE allocated by the compiler)
- 32-bit operand width
- Compile-time init values only (runtime init is straightforward; defer if not needed by target benchmarks)
- Magic-address-pattern PTX → PPTX lowering
- Simulator support for the PE config + bitstream encoding
- DICEwattch counter integration

### Out of scope (v2+)

- Data-dependent-address atomics (`atomicAdd(&hist[v], 1)`) — would need a separate "atomic-memory" SRAM structure or per-bank stateful slots
- 64-bit atomics (paired PEs) — defer until a benchmark needs them
- Cross-DBB accumulator state persistence — defer; v1 reads out + re-inits between DBBs
- Custom PTX opcode (vs magic-address Trojan horse) — defer to v2 if magic-address is too limiting
- Cross-CTA hierarchical accumulators — left to user code (per-CTA accumulator → 1 L2 atomic per CTA)

---

## 7. Benchmarks for evaluation

### Primary

| Benchmark | What it tests | DICE-side metric |
|---|---|---|
| **Pure block-sum** (microbenchmark) | Atomic-Add reduction | L2 atomics: `N_threads/32` → `1` per CTA |
| **Inclusive prefix scan** (microbenchmark) | Per-thread NEW state writeback | Replace huffman's Blelloch tree |
| **Compare-and-swap** (microbenchmark) | OLD-state semantics + first-arrival ordering | Replace spin-loop CAS pattern |

### Rodinia integration candidates

| Kernel | Atomic/scan use | Expected savings |
|---|---|---|
| `huffman` (scanLargeArray) | Inclusive scan via Blelloch | Replace 2-kernel tree+uniformAdd with 1 DBB accumulator |
| `hybridsort` (bucketprefixoffset) | Sequential per-thread prefix-sum | Direct port — same semantics, 1 PE |
| `streamcluster` (compute_cost) | Per-CTA cost reduction via shuffle+atomic | 1 atomic per CTA, no shuffle |
| `bfs` | Frontier compaction via atomic-or/cas | CAS-free first-arrival ordering |

### Comparison axes per benchmark

1. **Cycles** (DICE accumulator vs. GPU Blelloch/shuffle baseline)
2. **L2 atomic count** (DICE: 1 per CTA, GPU: 32+ per CTA)
3. **SM-dynamic energy** (DICEwattch report — RFP, SHRDP, SCHEDP, PIPEP)
4. **Total energy including L2/NoC** (DICEwattch — DRAMP, NOCP, L2CP)

---

## 8. Open questions

1. **Predication semantics for accumulator state update**: predicate gates only the writeback, or also the PE state update? Default proposal: gate both. Needs validation against real CUDA atomicAdd usage in control flow.

2. **Multi-accumulator placement**: when a kernel needs N accumulators, all in the same DBB, how does the placer handle PE pressure (16 PEs total, minus accumulators leaves fewer for compute)? Compiler may need to split into multiple DBBs.

3. **CAS retry semantics in source code**: do users write `do { ... } while (cas_failed)` and expect DICE to optimize the loop away (since DICE never has cas failure on first try)? Or do users use a different pattern? Probably worth a `dice_cta_acc_cas` that just does one try and returns OLD (no spin loop in source), with semantic documentation that on DICE the first try always succeeds if the value is current.

4. **64-bit and float accumulators**: PE width is 32-bit. atomicAdd on float should compile to FP-ALU mode of an accumulator PE. atomicCAS on 64-bit pointer needs paired PEs. Both deferred to v2.

5. **Bitstream encoding for the "accumulator mode" config bit**: probably a couple of bits in the existing PE config word. Need to confirm the bitstream has spare bits available.

---

## 9. Work plan

| # | Task |
|---|---|
| 1 | Write `dice_atomics.h` with the user-facing intrinsics (magic-address impl) |
| 2 | Write a microbenchmark `cuda/block_sum/block_sum.cu` (DICE accumulator version + reference SMEM-atomic version) |
| 3 | Add the magic-address recognition pass to the DICE p-graph compiler (the Python tool in `dice_ilp_compiler`) |
| 4 | Implement the `acc.{op}` PPTX instruction parser in `src/cuda-sim/dice_metadata.cc` |
| 5 | Implement the accumulator-mode PE config + crossbar-self-loop in `src/gpgpu-sim/cgra_core.cc` |
| 6 | Add `DICE_ACC_OPS_N` counter to DICEwattch + report integration |
| 7 | Run microbenchmarks end-to-end; compare DICE vs GPU at iso-config; report cycles + energy |
| 8 | Port one Rodinia kernel (start with `hybridsort/bucketprefixoffset` — simplest) |
| 9 | Write up results for ISCA / HPCA submission |
