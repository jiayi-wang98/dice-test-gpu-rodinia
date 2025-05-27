import re
import csv
import os
import argparse
from pathlib import Path
from datetime import datetime

def parse_log(log_path):
    patterns = {
        'kernel_push': re.compile(r"GPGPU-Sim PTX: pushing kernel '(?P<kernel>[^']+)' to stream \d+, gridDim= \((?P<grid>[^)]+)\) blockDim = \((?P<block>[^)]+)\)"),
        'kernel_name': re.compile(r"kernel_name = (?P<kernel_name>\S+)"),
        'kernel_launch_uid': re.compile(r"kernel_launch_uid = (?P<uid>\d+)"),
        'gpu_sim_cycle': re.compile(r"gpu_sim_cycle = (?P<gpu_sim_cycle>\d+)"),
        'gpu_tot_sim_cycle': re.compile(r"gpu_tot_sim_cycle = (?P<gpu_sim_cycle>\d+)"),
        'L2_BW': re.compile(r"L2_BW\s*=\s*(?P<L2_BW>\S+)"),
        'L2_BW_total': re.compile(r"L2_BW_total\s*=\s*(?P<L2_BW_total>\S+)"),
        'L1I_total_cache_accesses': re.compile(r"L1I_total_cache_accesses = (?P<L1I>\d+)"),
        'L1D_total_cache_accesses': re.compile(r"L1D_total_cache_accesses = (?P<L1D>\d+)"),
        'L1C_total_cache_accesses': re.compile(r"L1C_total_cache_accesses = (?P<L1C>\d+)"),
        'L1T_total_cache_accesses': re.compile(r"L1T_total_cache_accesses = (?P<L1T>\d+)"),
        'gpgpu_n_tot_regfile_acesses': re.compile(r"gpgpu_n_tot_regfile_acesses = (?P<regs>\d+)"),
        'L2_total_cache_accesses': re.compile(r"L2_total_cache_accesses = (?P<L2_total>\d+)")
    }

    entries = []
    current = None

    with open(log_path) as f:
        for line in f:
            m = patterns['kernel_push'].search(line)
            if m:
                if current:
                    entries.append(current)
                current = {
                    'kernel_name': m.group('kernel'),
                    'gridDim': m.group('grid'),
                    'BlockDim': m.group('block')
                }
                continue
            if not current:
                continue
            for key, pat in patterns.items():
                if key == 'kernel_push':
                    continue
                m = pat.search(line)
                if m:
                    try:
                        val = m.group(key)
                    except IndexError:
                        val = m.group(1)
                    current[key if key != 'kernel_name' else 'kernel_name'] = val
                    break
        if current:
            entries.append(current)
    return entries


def write_csv(entries, out_file, device_type):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cols = [
        'date_time', 'device_type',
        'kernel_name', 'kernel_launch_uid', 'gridDim', 'BlockDim',
        'gpu_sim_cycle', 'gpu_tot_sim_cycle', 'L2_BW', 'L2_BW_total',
        'L1I_total_cache_accesses', 'L1D_total_cache_accesses',
        'L1C_total_cache_accesses', 'L1T_total_cache_accesses',
        'gpgpu_n_tot_regfile_acesses', 'L2_total_cache_accesses'
    ]

    file_exists = out_file.exists()

    with open(out_file, 'a', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=cols)

        if not file_exists:
            writer.writeheader()

        for e in entries:
            row = {c: e.get(c, '') for c in cols[2:]}  # extract kernel fields only
            row['date_time'] = now
            row['device_type'] = device_type
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Parse GPGPU-Sim log and collect kernel stats')
    parser.add_argument('logfile', help='Path to log file')
    parser.add_argument('outdir', help='Directory to write results')
    parser.add_argument('--test_gpu', action='store_true', help='Label this run as GPU')
    parser.add_argument('--test_dice', action='store_true', help='Label this run as DICE')

    args = parser.parse_args()

    device_type = "GPU" if args.test_gpu else "DICE" if args.test_dice else "Unknown"

    log_path = Path(args.logfile)
    app = log_path.parent.name
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    entries = parse_log(log_path)
    out_file = outdir / f"{app}.result.csv"
    write_csv(entries, out_file, device_type)
    print(f"Wrote {len(entries)} entries to {out_file}")

if __name__ == '__main__':
    main()
