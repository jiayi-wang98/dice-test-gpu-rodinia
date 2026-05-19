#!/usr/bin/env python3
"""Compare DICEwattch SM-dynamic energy/power between RFU and SMEM-baseline
versions of each kernel. Reads paired reports saved as
  <bench_dir>/gpgpusim_dice_power_report_rfu.log
  <bench_dir>/gpgpusim_dice_power_report_smem.log

For multi-invocation kernels (none in this set so far), uses cycle-weighted
aggregation. For single-invocation kernels, per-kernel values are used directly.
"""
import re, sys, os

FREQ_GHZ = 1.47
DICE_ROOT = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/dice-test-gpu-rodinia/cuda"

# Per the user instruction, SM-dynamic excludes:
#   L2CP, MCP, NOCP, DRAMP, IDLE_COREP, CONST_DYNAMICP/CONSTP, STATICP, IBP
# Includes everything else.
SM_DYNAMIC_KEYS = [
    "ICP","DCP","TCP","CCP","SHRDP",
    "RFP","INTP","SPP","FPUP","DPUP","SFUP",
    "INT_MUL24P","INT_MUL32P","INT_MULP","INT_DIVP",
    "FP_MULP","FP_DIVP","FP_SQRTP","FP_LGP","FP_SINP","FP_EXP",
    "DP_MULP","DP_DIVP","TENSORP","TEXP",
    "SCHEDP","PIPEP","BCP",
]


def parse_report(path):
    """Return list of (kname, vals-dict) per kernel block."""
    if not os.path.exists(path):
        return []
    text = open(path).read()
    blocks = re.split(r"(?m)^kernel_name\s*=\s*", text)[1:]
    out = []
    for blk in blocks:
        m = re.match(r"(\S+)", blk)
        if not m:
            continue
        kname = m.group(1)
        avg = re.search(r"Kernel Average Power Data:(.*?)(?:Kernel Maximum|\Z)",
                        blk, re.S)
        if not avg:
            continue
        vals = {}
        for line in avg.group(1).splitlines():
            m2 = re.match(r"\s*(?:kernel_avg_power|gpu_avg_(\w+))[,\s]*=\s*"
                          r"([-\deE.+]+)", line)
            if not m2:
                continue
            key = m2.group(1) or "kernel_avg_power"
            try:
                vals[key] = float(m2.group(2))
            except ValueError:
                pass
        out.append((kname, vals))
    return out


def find_cycles(bench, log_glob):
    """Return list of (kname, cycles) from latest test_dice_*.log matching glob."""
    import glob
    candidates = sorted(glob.glob(log_glob))
    candidates = [c for c in candidates if "_stderr" not in c]
    if not candidates:
        return []
    out = []
    cur_kname = None
    with open(candidates[-1]) as f:
        for line in f:
            m = re.match(r"kernel_name\s*=\s*(\S+)", line)
            if m:
                cur_kname = m.group(1)
                continue
            m = re.match(r"gpu_sim_cycle\s*=\s*(\d+)", line)
            if m and cur_kname is not None:
                out.append((cur_kname, int(m.group(1))))
                cur_kname = None
    return out


def sm_dynamic(vals):
    return sum(vals.get(k, 0.0) for k in SM_DYNAMIC_KEYS)


def summarize(bench_dir, suffix):
    rpt = os.path.join(bench_dir, f"gpgpusim_dice_power_report_{suffix}.log")
    if not os.path.exists(rpt):
        return None
    powers = parse_report(rpt)
    if not powers:
        return None
    # Sum across all kernels in the report (for multi-kernel benches), and
    # also separate per-kernel for breakdown
    per_kernel = []
    for kname, vals in powers:
        sm_mw = sm_dynamic(vals)
        total_mw = vals.get("kernel_avg_power", 0.0)
        per_kernel.append({
            "kname":  kname,
            "sm_mw":  sm_mw,
            "tot_mw": total_mw,
            "RFP":    vals.get("RFP", 0.0),
            "SHRDP":  vals.get("SHRDP", 0.0),
            "DCP":    vals.get("DCP", 0.0),
            "SPP":    vals.get("SPP", 0.0) + vals.get("INTP", 0.0),
            "FPUP":   vals.get("FPUP", 0.0),
            "PIPEP":  vals.get("PIPEP", 0.0),
        })
    return per_kernel


def main():
    benches = ["stencil1d","stencil2d","stencil3d",
               "hotspot","pathfinder","backprop"]
    if len(sys.argv) > 1:
        benches = sys.argv[1:]

    rows = []
    for b in benches:
        bd = os.path.join(DICE_ROOT, b)
        rfu  = summarize(bd, "rfu")
        smem = summarize(bd, "smem")
        if not rfu or not smem:
            print(f"  {b}: MISSING (rfu={rfu is not None}, smem={smem is not None})")
            continue
        # Map smem kernels by short name (function-name suffix) for pairing
        import glob
        rfu_cycles  = dict(find_cycles(b, os.path.join(bd, "test_dice_*.log")))
        # We can't trivially separate rfu_cycles vs smem_cycles (the log was
        # overwritten). The smoke-test path runs SMEM second so the latest log
        # has SMEM cycles. RFU cycles come from kernel_max_power not the log.
        # For now just sum cycle×power across all kernels per side.
        for r, s in zip(rfu, smem):
            rows.append({
                "bench": b,
                "kname": r["kname"],
                "rfu":   r,
                "smem":  s,
            })

    # Pull per-kernel cycles from the _rfu_run.log and _smem_run.log.  For
    # multi-kernel benches (backprop) the log has multiple kernel blocks each
    # with their own gpu_sim_cycle line.
    def get_all_cycles(bench, suffix):
        import re
        log_path = os.path.join(DICE_ROOT, bench, f"_{suffix}_run.log")
        if not os.path.exists(log_path):
            return []
        out = []
        with open(log_path) as f:
            for line in f:
                m = re.match(r"gpu_sim_cycle\s*=\s*(\d+)", line)
                if m:
                    out.append(int(m.group(1)))
        return out

    # Build per-bench cycle index so kernel #i gets its own cycle count
    rfu_cycles_by_bench  = {b: get_all_cycles(b, "rfu")  for b in benches}
    smem_cycles_by_bench = {b: get_all_cycles(b, "smem") for b in benches}
    # Track kernel-index per bench as we walk the rows
    kernel_idx = {}
    def get_cycles(bench, suffix):
        i = kernel_idx.setdefault(f"{bench}/{suffix}", 0)
        lst = (rfu_cycles_by_bench if suffix == "rfu" else smem_cycles_by_bench)[bench]
        if i >= len(lst):
            return 0
        kernel_idx[f"{bench}/{suffix}"] = i + 1
        return lst[i]

    # Per-kernel summary: total energy = power × cycles / freq, in nJ.
    # For multi-kernel benches we sum per-kernel energies and pair by index.
    print(f"\n{'Bench':<11} {'Kernel':<30} | "
          f"{'SMEM cyc':>8} {'RFU cyc':>8} {'Δcyc%':>6} | "
          f"{'SMEM E':>9} {'RFU E':>9} {'ΔE%':>6} | "
          f"{'SMEM SHRD E':>11} {'RFU SHRD E':>11} | "
          f"{'SMEM RF E':>10} {'RFU RF E':>10}")
    print("-" * 150)
    for row in rows:
        b = row["bench"]; r = row["rfu"]; s = row["smem"]
        kn = row["kname"][:30]
        r_cyc = get_cycles(b, "rfu")
        s_cyc = get_cycles(b, "smem")
        # Energy nJ = power_mW × cycles / freq_GHz × 1e-3? But the wrapper's
        # mW values are computed as e_nJ × freq / cycles, so reversing:
        # total_e_nJ = mW × cycles / freq.  Same proportional relationship
        # regardless of mW vs W interpretation.
        def E(mw, cyc):
            return mw * cyc / FREQ_GHZ
        s_E    = E(s["sm_mw"],  s_cyc)
        r_E    = E(r["sm_mw"],  r_cyc)
        s_RF_E = E(s["RFP"],    s_cyc)
        r_RF_E = E(r["RFP"],    r_cyc)
        s_SH_E = E(s["SHRDP"],  s_cyc)
        r_SH_E = E(r["SHRDP"],  r_cyc)
        dcyc = (s_cyc - r_cyc) / s_cyc * 100 if s_cyc > 0 else 0
        dE   = (s_E - r_E) / s_E * 100      if s_E > 0   else 0
        print(f"{b:<11} {kn:<30} | "
              f"{s_cyc:>8d} {r_cyc:>8d} {dcyc:>5.1f}% | "
              f"{s_E:>9.0f} {r_E:>9.0f} {dE:>5.1f}% | "
              f"{s_SH_E:>11.1f} {r_SH_E:>11.1f} | "
              f"{s_RF_E:>10.1f} {r_RF_E:>10.1f}")


if __name__ == "__main__":
    main()
