// dice_atomics.h — magic-address Trojan-horse intrinsics for DICE CGRA
// accumulator PE.
//
// Source-level call:    dice_cta_acc_add(X, val);
// Compiles to PTX:      atom.shared.add.u32 r_dst, [&__dice_acc_slots[slot]], r_val
// DICE compiler/sim:    recognizes addr ∈ __dice_acc_slots[] at lowering, swaps
//                       the L2-bound SMEM atomic for a PE in accumulator-mode.
// Vanilla GPU build:    falls through to a real SMEM atomicAdd — exact
//                       functional equivalent so the source compiles unchanged
//                       on NVCC + runs on an unmodified driver.
//
// Variable name == accumulator identity. Each distinct variable used as the
// first argument across a kernel binds to one PE in accumulator mode. The
// compiler picks slot indices; here we just spread them by source-order.

#ifndef DICE_ATOMICS_H
#define DICE_ATOMICS_H

// Reserved SMEM region the DICE compiler watches for. Treated as a magic
// handle on DICE (no real storage allocated); on vanilla GPUs it is real
// SMEM that the atomicAdd writes through. Fixed-size to keep the PTX form
// as `.shared .align N .b8 ...` (no `.extern`), which the DICE parser
// accepts unchanged.
//
// Slot count caps the number of distinct accumulators per CTA; 32 is more
// than enough for the patterns we care about (block reduce, scan).
#ifndef DICE_ACC_SLOTS
#define DICE_ACC_SLOTS 32
#endif

__shared__ unsigned int __dice_acc_slots[DICE_ACC_SLOTS];

// Programmer's view: the X variable IS the accumulator. The PE writeback
// lands in X's register; on the GPU baseline the same source code falls
// through to a real SMEM atomicAdd that produces the same numeric result.
//
// The SLOT template parameter only exists to disambiguate *multiple*
// logical accumulators in the same kernel at the PTX level — the DICE
// compiler reads the address inside `[&__dice_acc_slots[SLOT]]` to decide
// which PE to bind. Programmers shouldn't have to manage it: the
// `dice_cta_acc_add` macro at the bottom of this header auto-assigns a
// unique slot per source-level call site via __COUNTER__.
//
// For the rare case where the same logical accumulator is fired from two
// different call sites (e.g., inside two arms of an if/else), use the
// explicit-slot form `dice_cta_acc_add_slot<N>(X, val)` with a shared N
// so both sites bind to the same PE.

template <unsigned SLOT>
__device__ __forceinline__ unsigned int
dice_cta_acc_add_slot(unsigned int &X, unsigned int val) {
    static_assert(SLOT < DICE_ACC_SLOTS, "slot out of range");
    unsigned int old = atomicAdd(&__dice_acc_slots[SLOT], val);
    // GPU baseline returns OLD; DICE add-accumulator returns NEW (inclusive
    // scan). Reconstruct NEW on the GPU side so both paths match.
    X = old + val;
    return X;
}

template <unsigned SLOT>
__device__ __forceinline__ int
dice_cta_acc_add_slot(int &X, int val) {
    static_assert(SLOT < DICE_ACC_SLOTS, "slot out of range");
    int old = (int)atomicAdd((int *)&__dice_acc_slots[SLOT], val);
    X = old + val;
    return X;
}

// Macro form — automatic slot assignment via __COUNTER__.
//
// Each call site in the translation unit expands with a fresh integer, so
// `dice_cta_acc_add(sum, v)` and `dice_cta_acc_add(prod, v)` get distinct
// slots without any programmer-facing identifier. The macro is variadic so
// it transparently wraps both the (int&, int) and (unsigned&, unsigned)
// overloads.
//
// Modulo DICE_ACC_SLOTS guards against TUs that consume so many counters
// (other libraries also use __COUNTER__) that the raw index would exceed
// the slot array — the static_assert in dice_cta_acc_add_slot still
// catches out-of-range explicit usage.
#define dice_cta_acc_add(...) \
    dice_cta_acc_add_slot<((__COUNTER__) % DICE_ACC_SLOTS)>(__VA_ARGS__)

// ===================================================================
// MAX-mode intrinsic: same magic-address trick, atomic.shared.max.
// The state-PE configured in MAX mode retains the running maximum
// across dispatched threads; the OLD value is returned (= the max
// observed by this thread before its own update).
//
// On GPU baseline: lowers to atom.shared.max which is sequential at
// the SMEM atomic port (same as atomicAdd).
// On DICE: routes to the state-PE in MAX mode via the same magic-
// address recognition; the per-op energy is PIPE_A (same as ADD).
// ===================================================================
template <unsigned SLOT>
__device__ __forceinline__ int
dice_cta_acc_max_slot(int &X, int val) {
    static_assert(SLOT < DICE_ACC_SLOTS, "slot out of range");
    int old = (int)atomicMax((int *)&__dice_acc_slots[SLOT], val);
    X = (old > val) ? old : val;   // NEW = max(old, val)
    return X;
}

template <unsigned SLOT>
__device__ __forceinline__ unsigned int
dice_cta_acc_max_slot(unsigned int &X, unsigned int val) {
    static_assert(SLOT < DICE_ACC_SLOTS, "slot out of range");
    unsigned int old = atomicMax(&__dice_acc_slots[SLOT], val);
    X = (old > val) ? old : val;
    return X;
}

#define dice_cta_acc_max(...) \
    dice_cta_acc_max_slot<((__COUNTER__) % DICE_ACC_SLOTS)>(__VA_ARGS__)

// MIN-mode intrinsic (mirror of MAX).
template <unsigned SLOT>
__device__ __forceinline__ int
dice_cta_acc_min_slot(int &X, int val) {
    static_assert(SLOT < DICE_ACC_SLOTS, "slot out of range");
    int old = (int)atomicMin((int *)&__dice_acc_slots[SLOT], val);
    X = (old < val) ? old : val;
    return X;
}

template <unsigned SLOT>
__device__ __forceinline__ unsigned int
dice_cta_acc_min_slot(unsigned int &X, unsigned int val) {
    static_assert(SLOT < DICE_ACC_SLOTS, "slot out of range");
    unsigned int old = atomicMin(&__dice_acc_slots[SLOT], val);
    X = (old < val) ? old : val;
    return X;
}

#define dice_cta_acc_min(...) \
    dice_cta_acc_min_slot<((__COUNTER__) % DICE_ACC_SLOTS)>(__VA_ARGS__)

// ===================================================================
// FMA-mode loop-carry intrinsic for first-order linear recurrences
//   y_new = a*x + b*y_prev
//
// Used for IIR filters, EMA smoothing, any 1st-order LTI recurrence.
// On DICE: routes to a switch-box-configured feedback path through
// an FMA-mode PE.  No new hardware - the CGRA switch boxes already
// have configurable bypass registers; the bitstream toggles the
// feedback wire on for the configured slot.
// On GPU baseline: this intrinsic is not used (separate iir.cu
// uses the standard per-thread serial loop pattern).
//
// Implementation here: read slot, compute FMA, atomicExch the new
// value.  Functionally correct under DICE's dispatch-order serial-
// isation; would race on a stock GPU SIMT (which is why the GPU
// baseline uses a different kernel structure).
// ===================================================================
template <unsigned SLOT>
__device__ __forceinline__ float
dice_loop_carry_fma_slot(float &y_out, float x, float a, float b) {
    static_assert(SLOT < DICE_ACC_SLOTS, "slot out of range");
    int *slot_p = (int *)&__dice_acc_slots[SLOT];
    int old_int, new_int;
    float y_prev, y_new;
    // CAS-loop: each iteration atomically reads-modifies-writes the slot.
    // On a stock GPU this is the standard retry-until-CAS-succeeds idiom
    // for float reductions (no native atomicFMA in PTX).
    // On DICE the compiler can recognise this magic-address CAS-loop and
    // lower it to a single-cycle in-fabric FMA dispatch through the
    // configured switch-box feedback path.
    do {
        old_int = atomicAdd(slot_p, 0);                // atomic read of slot
        y_prev  = __int_as_float(old_int);
        y_new   = a * x + b * y_prev;
        new_int = __float_as_int(y_new);
    } while (atomicCAS(slot_p, old_int, new_int) != old_int);
    y_out = y_new;
    return y_prev;
}

#define dice_loop_carry_fma(...) \
    dice_loop_carry_fma_slot<((__COUNTER__) % DICE_ACC_SLOTS)>(__VA_ARGS__)

#endif  // DICE_ATOMICS_H
