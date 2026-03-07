import re
import os
import csv
import argparse
from collections import defaultdict, OrderedDict

# Metrics you want as the left-most column (row order preserved)
METRICS = [
    "kernel_avg_power",
    "gpu_avg_IBP",
    "gpu_avg_ICP",
    "gpu_avg_DCP",
    "gpu_avg_TCP",
    "gpu_avg_CCP",
    "gpu_avg_SHRDP",
    "gpu_avg_RFP",
    "gpu_avg_INTP",
    "gpu_avg_FPUP",
    "gpu_avg_DPUP",
    "gpu_avg_INT_MUL24P",
    "gpu_avg_INT_MUL32P",
    "gpu_avg_INT_MULP",
    "gpu_avg_INT_DIVP",
    "gpu_avg_FP_MULP",
    "gpu_avg_FP_DIVP",
    "gpu_avg_FP_SQRTP",
    "gpu_avg_FP_LGP",
    "gpu_avg_FP_SINP",
    "gpu_avg_FP_EXP",
    "gpu_avg_DP_MULP",
    "gpu_avg_DP_DIVP",
    "gpu_avg_TENSORP",
    "gpu_avg_TEXP",
    "gpu_avg_SCHEDP",
    "gpu_avg_L2CP",
    "gpu_avg_MCP",
    "gpu_avg_NOCP",
    "gpu_avg_DRAMP",
    "gpu_avg_PIPEP",
    "gpu_avg_IDLE_COREP",
    "gpu_avg_CONSTP",
    "gpu_avg_STATICP",
]

METRIC_SET = set(METRICS)

KERNEL_NAME_RE = re.compile(r"^\s*kernel_name\s*=\s*(\S+)\s*$")
KERNEL_UID_RE  = re.compile(r"^\s*kernel_launch_uid\s*=\s*(\d+)\s*$")
AVG_SECTION_RE = re.compile(r"^\s*Kernel Average Power Data:\s*$")

# Matches lines like:
#   kernel_avg_power = 77.4611
#   gpu_avg_IBP, = 0.0425198
#   gpu_avg_CONSTP = 32.3252
METRIC_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*,?\s*=\s*([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*$"
)

def safe_filename(s: str) -> str:
    # Keep it readable but filesystem-safe
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)

def parse_log(path: str):
    """
    Returns:
      kernel_runs: dict[kernel_name] -> list of dict{metric -> value (float)}
      kernel_uids: dict[kernel_name] -> list of launch_uid strings (optional, for headers)
    """
    kernel_runs = defaultdict(list)
    kernel_uids = defaultdict(list)

    current_kernel = None
    current_uid = None
    in_avg_section = False
    current_run = None

    with open(path, "r", errors="ignore") as f:
        for line in f:
            # New kernel block starts
            m = KERNEL_NAME_RE.match(line)
            if m:
                current_kernel = m.group(1)
                current_uid = None
                in_avg_section = False
                current_run = None
                continue

            m = KERNEL_UID_RE.match(line)
            if m:
                current_uid = m.group(1)
                continue

            # Enter average-power section
            if AVG_SECTION_RE.match(line):
                if current_kernel is None:
                    # If log is malformed, skip
                    continue
                in_avg_section = True
                current_run = {}
                continue

            if in_avg_section:
                # End conditions: next section begins, or another kernel appears, etc.
                if line.strip() == "" or line.startswith("Kernel Maximum Power Data:") or line.startswith("Kernel Minimum Power Data:") or line.startswith("Accumulative Power Statistics"):
                    # Finalize this run (even if partially filled)
                    if current_run is not None:
                        kernel_runs[current_kernel].append(current_run)
                        kernel_uids[current_kernel].append(current_uid if current_uid is not None else str(len(kernel_runs[current_kernel])))
                    in_avg_section = False
                    current_run = None
                    continue

                mm = METRIC_LINE_RE.match(line)
                if mm:
                    key = mm.group(1).strip()  # commas already handled by regex
                    if key in METRIC_SET:
                        val = float(mm.group(2))
                        current_run[key] = val

    # If file ended while still in avg section, flush last run
    if in_avg_section and current_run is not None and current_kernel is not None:
        kernel_runs[current_kernel].append(current_run)
        kernel_uids[current_kernel].append(current_uid if current_uid is not None else str(len(kernel_runs[current_kernel])))

    return kernel_runs, kernel_uids

def write_kernel_csvs(kernel_runs, kernel_uids, out_dir: str, no_metric_column: bool):
    os.makedirs(out_dir, exist_ok=True)

    for kname, runs in kernel_runs.items():
        if not runs:
            continue

        # Column headers
        if no_metric_column:
            headers = []
        else:
            headers = ["metric"]

        for i, uid in enumerate(kernel_uids.get(kname, []), start=1):
            headers.append(f"run{i}_uid{uid}")

        rows = []
        for metric in METRICS:
            row = []

            if not no_metric_column:
                row.append(metric)

            for r in runs:
                row.append("" if metric not in r else f"{r[metric]}")

            rows.append(row)

        out_path = os.path.join(out_dir, f"{safe_filename(kname)}_power.csv")
        with open(out_path, "w", newline="") as wf:
            w = csv.writer(wf)
            w.writerow(headers)
            w.writerows(rows)

        print(f"Wrote: {out_path}  (runs={len(runs)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Path to the .log file")
    ap.add_argument("--out", default=".", help="Output directory for <kernel>_power.csv files")
    ap.add_argument("--no-metric-column", action="store_true",
                    help="Do not print the leftmost metric name column")
    args = ap.parse_args()

    kernel_runs, kernel_uids = parse_log(args.logfile)
    write_kernel_csvs(kernel_runs, kernel_uids, args.out, args.no_metric_column)

if __name__ == "__main__":
    main()