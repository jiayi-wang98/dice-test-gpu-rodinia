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
// Float ADD-mode overload. CAS-loop on the bit-cast slot (CUDA's
// atomicAdd on float exists for global memory but at this SMEM-magic-
// address path we go through the same int-cast CAS pattern as the
// FMA loop-carry intrinsic below so the lowering pass sees a uniform
// IR shape regardless of element type.
template <unsigned SLOT>
__device__ __forceinline__ float
dice_cta_acc_add_slot(float &X, float val) {
    static_assert(SLOT < DICE_ACC_SLOTS, "slot out of range");
    int *slot_p = (int *)&__dice_acc_slots[SLOT];
    int old_int, new_int;
    float old_f, new_f;
    do {
        old_int = atomicAdd(slot_p, 0);
        old_f   = __int_as_float(old_int);
        new_f   = old_f + val;
        new_int = __float_as_int(new_f);
    } while (atomicCAS(slot_p, old_int, new_int) != old_int);
    X = new_f;
    return new_f;
}

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

// Sortable-int encoding for float MAX-mode: bijectively maps a
// float to a signed int whose ordering matches the float ordering,
// letting us use a single atomicMax.s32 (with PTX atom.shared.max.s32)
// instead of a CAS-loop.  Avoids the data-dependent early-out branch
// that nvcc reintroduces when the CAS-loop's new value happens to
// equal the old value (which would deadlock the DICE SIMT stack when
// chained with a downstream CAS-loop, see softmax_acc).
//
// Encoding: keep the sign bit, flip the mantissa+exponent bits for
// negatives.  Standard radix-sort-for-floats trick.  Self-inverse.
__device__ __forceinline__ int dice_float_to_sortable(float f) {
    int x = __float_as_int(f);
    int mask = (x >> 31) & 0x7FFFFFFF;
    return x ^ mask;
}
__device__ __forceinline__ float dice_sortable_to_float(int s) {
    int mask = (s >> 31) & 0x7FFFFFFF;
    return __int_as_float(s ^ mask);
}
// Init sentinel for a MAX slot: the sortable-int encoding of -FLT_MAX.
// (NOT INT_MIN -- INT_MIN decodes to NaN under this encoding because
// the lower 31 bits flip to all-1s when XOR'd with 0x7FFFFFFF, giving
// the float NaN bit pattern.  For benchmarks that only DECODE the
// slot after every thread has fired (e.g.\ block_max, the 3-pass
// softmax), INT_MIN happens to work because the slot always holds a
// real encoded xi by the time it's read.  For benchmarks where a
// thread may read the slot before any update (online softmax: the
// first thread's m_old is the sentinel), the decode round-trip
// matters and we use encoded(-FLT_MAX) = 0x80800000.)
#define DICE_ACC_MAX_INIT_F  (0x80800000)

// Float MAX-mode overload.  Single atomicMax on the sortable encoding;
// kernel author must initialise the slot to DICE_ACC_MAX_INIT_F (NOT
// to -FLT_MAX) before the first call.
template <unsigned SLOT>
__device__ __forceinline__ float
dice_cta_acc_max_slot(float &X, float val) {
    static_assert(SLOT < DICE_ACC_SLOTS, "slot out of range");
    int *slot_p   = (int *)&__dice_acc_slots[SLOT];
    int  s_val    = dice_float_to_sortable(val);
    int  s_old    = atomicMax(slot_p, s_val);
    int  s_new    = (s_old > s_val) ? s_old : s_val;
    X = dice_sortable_to_float(s_new);
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

// ===================================================================
// Coupled-slot IFF: FlashAttention-style online softmax update.
//
// Single-pass (m, l) maintenance across threads in CTA-dispatch order:
//     m_new = max(m_old, x)
//     l_new = l_old * exp(m_old - m_new) + exp(x - m_new)
//     slot_M := m_new
//     slot_L := l_new
//
// The L-slot's update consumes BOTH m_old (slot-M before this thread's
// update) and m_new (slot-M after).  On the CGRA fabric this is a
// cross-slot wire: the M-PE's output is routed through the switch
// box into the L-PE's input in the same dispatch cycle, allowing
// the (m, l) tuple to advance together per thread.  This is the
// canonical FlashAttention online softmax algorithm fused into a
// single IFF p-graph (instead of separate MAX-pass + ADD-pass).
//
// SEMANTICS: correct only under DICE's CTA-order dispatch.  Stock
// GPU SIMT cannot soundly run this kernel as written (m_old/m_new
// across the two atom ops race against other warps).  We provide
// this intrinsic specifically as the DICE-only AI primitive.
// ===================================================================
template <unsigned SLOT_M, unsigned SLOT_L>
__device__ __forceinline__ void
dice_online_softmax_update_slot(float x) {
    static_assert(SLOT_M < DICE_ACC_SLOTS && SLOT_L < DICE_ACC_SLOTS,
                  "slots out of range");
    static_assert(SLOT_M != SLOT_L, "M and L slots must differ");
    int *m_p = (int *)&__dice_acc_slots[SLOT_M];
    int *l_p = (int *)&__dice_acc_slots[SLOT_L];

    // M update: sortable-int atomic max (returns OLD, slot ← max).
    int x_sortable = dice_float_to_sortable(x);
    int s_old = atomicMax(m_p, x_sortable);
    int s_new = (s_old > x_sortable) ? s_old : x_sortable;
    float m_old = dice_sortable_to_float(s_old);
    float m_new = dice_sortable_to_float(s_new);

    // L update: CAS-loop on (l_old, l_new). Under DICE dispatch-order
    // each thread's CAS succeeds first try (no concurrent writers).
    int   l_old_int, l_new_int;
    float l_old, l_new;
    do {
        l_old_int = atomicAdd(l_p, 0);             // atomic read of L
        l_old     = __int_as_float(l_old_int);
        float scale    = __expf(m_old - m_new);    // rescale running sum
        float new_term = __expf(x     - m_new);    // contribution from this thread
        l_new     = l_old * scale + new_term;
        l_new_int = __float_as_int(l_new);
    } while (atomicCAS(l_p, l_old_int, l_new_int) != l_old_int);
}

// Macro form pinning slots 0,1 by default (caller can use the
// _slot template directly to pick others).
#define dice_online_softmax_update(x) \
    dice_online_softmax_update_slot<0, 1>(x)

// Init sentinel pair for the (M, L) slot pair before the IFF chain.
#define DICE_ONLINE_SOFTMAX_INIT(slots)                          \
    do {                                                         \
        ((int   *)(slots))[0] = DICE_ACC_MAX_INIT_F;             \
        ((float *)(slots))[1] = 0.0f;                            \
    } while (0)

#endif  // DICE_ATOMICS_H
