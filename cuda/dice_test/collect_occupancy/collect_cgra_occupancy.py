import re
import csv
import os
from collections import defaultdict, OrderedDict

def parse_occupancy_distributions(log_file_path):
    """
    Parse log file and extract all occupancy distribution data.
    
    Args:
        log_file_path (str): Path to the input log file
        
    Returns:
        dict: Dictionary containing all parsed occupancy data organized by kernel
    """
    
    # Dictionary to store all kernel data
    kernels_data = {}
    kernel_names = {}
    
    # Regular expressions for parsing
    kernel_name_pattern = r'kernel_name\s*=\s*(.+?)(?=\n|\r|$)'
    kernel_uid_pattern = r'kernel_launch_uid\s*=\s*(\d+)'
    
    # Patterns for different occupancy distributions
    # These patterns stop at the next section or at a blank line followed by non-occupancy content
    pe_occupancy_pattern = r'DICE dispatcher Total occupancy distribution:(.*?)(?=\n\nDICE dispatcher CGRA PE|\n\nkernel_name|\n\nkernel_launch_uid|\n\n[A-Z]|\Z)'
    cgra_occupancy_pattern = r'DICE dispatcher CGRA PE occupancy distribution:(.*?)(?=\n\nDICE dispatcher LDST|\n\nkernel_name|\n\nkernel_launch_uid|\n\n[A-Z]|\Z)'
    ldst_occupancy_pattern = r'DICE dispatcher LDST Unit occupancy distribution:(.*?)(?=\n\nDICE dispatcher branch|\n\nkernel_name|\n\nkernel_launch_uid|\n\n[A-Z]|\Z)'
    branch_occupancy_pattern = r'DICE dispatcher branch occupancy distribution:(.*?)(?=\n\n|\n\nkernel_name|\n\nkernel_launch_uid|\n\nDICE dispatcher Total|\n\n[A-Z]|\Z)'
    
    try:
        with open(log_file_path, 'r') as file:
            content = file.read()
            
        # Find all kernel names and UIDs
        kernel_name_matches = list(re.finditer(kernel_name_pattern, content))
        kernel_uid_matches = list(re.finditer(kernel_uid_pattern, content))
        
        # Find all occupancy distributions
        pe_matches = list(re.finditer(pe_occupancy_pattern, content, re.DOTALL))
        cgra_matches = list(re.finditer(cgra_occupancy_pattern, content, re.DOTALL))
        ldst_matches = list(re.finditer(ldst_occupancy_pattern, content, re.DOTALL))
        branch_matches = list(re.finditer(branch_occupancy_pattern, content, re.DOTALL))
        
        # Convert to lists with positions
        name_positions = [(match.group(1).strip(), match.start()) for match in kernel_name_matches]
        uid_positions = [(match.group(1), match.start()) for match in kernel_uid_matches]
        
        # Process PE occupancy distributions
        for match in pe_matches:
            data = match.group(1)
            position = match.start()
            
            # Find matching UID and name
            matching_uid = find_matching_uid(position, uid_positions)
            matching_name = find_matching_name(position, name_positions)
            
            if matching_uid:
                if matching_uid not in kernels_data:
                    kernels_data[matching_uid] = {}
                    kernel_names[matching_uid] = matching_name or f"Unknown_Kernel_{matching_uid}"
                
                # Parse PE occupancy data
                pe_data = parse_pe_occupancy(data)
                kernels_data[matching_uid].update(pe_data)
        
        # Process CGRA occupancy distributions
        for match in cgra_matches:
            data = match.group(1)
            position = match.start()
            
            matching_uid = find_matching_uid(position, uid_positions)
            
            if matching_uid and matching_uid in kernels_data:
                cgra_data = parse_cgra_occupancy(data)
                kernels_data[matching_uid].update(cgra_data)
        
        # Process LDST occupancy distributions
        for match in ldst_matches:
            data = match.group(1)
            position = match.start()
            
            matching_uid = find_matching_uid(position, uid_positions)
            
            if matching_uid and matching_uid in kernels_data:
                ldst_data = parse_ldst_occupancy(data)
                kernels_data[matching_uid].update(ldst_data)
        
        # Process Branch occupancy distributions
        for match in branch_matches:
            data = match.group(1)
            position = match.start()
            
            matching_uid = find_matching_uid(position, uid_positions)
            
            if matching_uid and matching_uid in kernels_data:
                branch_data = parse_branch_occupancy(data)
                kernels_data[matching_uid].update(branch_data)
    
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        return {}, {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}, {}
    
    return kernels_data, kernel_names

def find_matching_uid(position, uid_positions):
    """Find the most recent kernel UID before the given position."""
    matching_uid = None
    for uid, uid_pos in uid_positions:
        if uid_pos < position:
            matching_uid = uid
        else:
            break
    return matching_uid

def find_matching_name(position, name_positions):
    """Find the most recent kernel name before the given position."""
    matching_name = None
    for name, name_pos in name_positions:
        if name_pos < position:
            matching_name = name
        else:
            break
    return matching_name

def parse_pe_occupancy(data):
    """Parse Total occupancy data including Stall metrics."""
    parsed = {}
    lines = data.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line looks like occupancy data
        if ':' in line and not line.startswith('Cache_') and not line.startswith('BW_'):
            key, value = line.split(':', 1)
            key = key.strip()
            
            # Accept Stall metrics
            if key.startswith('Stall'):
                try:
                    parsed[f"Total_{key}"] = int(value.strip())
                except ValueError:
                    print(f"Warning: Could not parse value '{value}' for metric '{key}'")
                    parsed[f"Total_{key}"] = 0
            # Accept Total followed by a number
            elif key.startswith('Total') and key[5:].isdigit():
                try:
                    parsed[f"Total_{key}"] = int(value.strip())
                except ValueError:
                    print(f"Warning: Could not parse value '{value}' for metric '{key}'")
                    parsed[f"Total_{key}"] = 0
    
    return parsed

def parse_cgra_occupancy(data):
    """Parse CGRA occupancy data."""
    parsed = {}
    lines = data.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            
            # Only accept PE followed by a number
            if key.startswith('PE') and key[2:].isdigit():
                try:
                    parsed[f"CGRA_{key}"] = int(value.strip())
                except ValueError:
                    print(f"Warning: Could not parse value '{value}' for metric '{key}'")
                    parsed[f"CGRA_{key}"] = 0
    
    return parsed

def parse_ldst_occupancy(data):
    """Parse LDST Unit occupancy data."""
    parsed = {}
    lines = data.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if ':' in line and 'LDST Unit' in line:
            key, value = line.split(':', 1)
            key = key.strip().replace(' ', '_')
            
            try:
                parsed[key] = int(value.strip())
            except ValueError:
                print(f"Warning: Could not parse value '{value}' for metric '{key}'")
                parsed[key] = 0
    
    return parsed

def parse_branch_occupancy(data):
    """Parse Branch Unit occupancy data."""
    parsed = {}
    lines = data.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if ':' in line and 'Branch Unit' in line:
            key, value = line.split(':', 1)
            key = key.strip().replace(' ', '_')
            
            try:
                parsed[key] = int(value.strip())
            except ValueError:
                print(f"Warning: Could not parse value '{value}' for metric '{key}'")
                parsed[key] = 0
    
    return parsed

def calculate_individual_kernel_data(kernels_data):
    """
    Convert cumulative kernel data to individual kernel data by subtracting previous kernels.
    
    Args:
        kernels_data (dict): Dictionary with kernel UIDs as keys and cumulative data as values
        
    Returns:
        dict: Dictionary with individual kernel data
    """
    # Sort kernel UIDs numerically to process in order
    sorted_kernel_uids = sorted(kernels_data.keys(), key=int)
    
    # Get all unique metric names
    all_metrics = set()
    for data in kernels_data.values():
        all_metrics.update(data.keys())
    
    individual_data = {}
    previous_cumulative = {}
    
    for uid in sorted_kernel_uids:
        individual_data[uid] = {}
        current_cumulative = kernels_data[uid]
        
        for metric in all_metrics:
            current_value = current_cumulative.get(metric, 0)
            previous_value = previous_cumulative.get(metric, 0)
            
            # Individual kernel value = current cumulative - previous cumulative
            individual_value = current_value - previous_value
            individual_data[uid][metric] = individual_value
        
        # Update previous cumulative for next iteration
        previous_cumulative = current_cumulative.copy()
    
    return individual_data

def get_ordered_metrics(all_metrics):
    """
    Order metrics in a logical way for CSV output.
    
    Args:
        all_metrics (set): Set of all metric names
        
    Returns:
        list: Ordered list of metrics
    """
    ordered = []
    
    # First, add Stall metrics from Total section
    stall_metrics = sorted([m for m in all_metrics if m.startswith('Total_Stall')])
    ordered.extend(stall_metrics)
    
    # Then add Total metrics (Total1 to Total93)
    total_metrics = []
    for i in range(94):  # Total0 to Total93
        metric = f'Total_Total{i}'
        if metric in all_metrics:
            total_metrics.append(metric)
    ordered.extend(total_metrics)
    
    # Then add CGRA metrics (CGRA_PE0 to CGRA_PE64)
    cgra_metrics = []
    for i in range(65):  # CGRA_PE0 to CGRA_PE64
        metric = f'CGRA_PE{i}'
        if metric in all_metrics:
            cgra_metrics.append(metric)
    ordered.extend(cgra_metrics)
    
    # Then add LDST Unit metrics
    ldst_metrics = []
    for i in range(9):  # LDST Unit 0 to 8
        metric = f'LDST_Unit_{i}'
        if metric in all_metrics:
            ldst_metrics.append(metric)
    ordered.extend(ldst_metrics)
    
    # Finally add Branch Unit metrics
    branch_metrics = []
    for i in range(5):  # Branch Unit 0 to 4
        metric = f'Branch_Unit_{i}'
        if metric in all_metrics:
            branch_metrics.append(metric)
    ordered.extend(branch_metrics)
    
    # Add any remaining metrics that weren't caught
    remaining = sorted(all_metrics - set(ordered))
    ordered.extend(remaining)
    
    return ordered

def write_to_csv(output_csv_path, log_file_name, kernels_data, kernel_names):
    """
    Write or append the parsed kernel data to CSV file.
    Groups kernels by name and outputs summary for each unique kernel name.
    
    Args:
        output_csv_path (str): Path to output CSV file
        log_file_name (str): Name of the log file being processed
        kernels_data (dict): Dictionary with kernel UIDs as keys and parsed data as values
        kernel_names (dict): Dictionary mapping kernel UIDs to kernel names
    """
    if not kernels_data:
        print(f"No kernel data found in {log_file_name}")
        return
    
    # Convert cumulative data to individual kernel data
    individual_data = calculate_individual_kernel_data(kernels_data)
    
    # Group kernels by name and sum their individual values
    kernel_summaries = {}
    kernel_counts = {}
    
    for uid, data in individual_data.items():
        kernel_name = kernel_names.get(uid, f"Unknown_{uid}")
        
        if kernel_name not in kernel_summaries:
            kernel_summaries[kernel_name] = {}
            kernel_counts[kernel_name] = 0
        
        kernel_counts[kernel_name] += 1
        
        # Sum the metrics for kernels with the same name
        for metric, value in data.items():
            if metric not in kernel_summaries[kernel_name]:
                kernel_summaries[kernel_name][metric] = 0
            kernel_summaries[kernel_name][metric] += value
    
    # Get all metrics and order them
    all_metrics = set()
    for summary in kernel_summaries.values():
        all_metrics.update(summary.keys())
    
    ordered_metrics = get_ordered_metrics(all_metrics)
    
    # Sort kernel names for consistent ordering
    sorted_kernel_names = sorted(kernel_summaries.keys())
    
    # Create column names with log file prefix and kernel count info
    column_names = []
    for kernel_name in sorted_kernel_names:
        count = kernel_counts[kernel_name]
        # Create a unique column name that includes the log file name
        col_name = f"{os.path.splitext(log_file_name)[0]}_{kernel_name}_sum{count}"
        column_names.append(col_name)
    
    # Check if CSV exists
    csv_exists = os.path.exists(output_csv_path)
    
    # Write or append to CSV
    if not csv_exists:
        # Create new CSV file
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header with empty first cell
            header = [''] + column_names
            writer.writerow(header)
            
            # Write data rows
            for metric in ordered_metrics:
                row = [metric]
                for kernel_name in sorted_kernel_names:
                    value = kernel_summaries[kernel_name].get(metric, 0)
                    row.append(value)
                writer.writerow(row)
    else:
        # Append to existing CSV
        # Read all existing rows
        with open(output_csv_path, 'r', newline='', encoding='utf-8') as readfile:
            reader = csv.reader(readfile)
            rows = list(reader)
        
        if not rows:
            # File exists but is empty, treat as new
            rows = [[''] + column_names]
            for metric in ordered_metrics:
                row = [metric]
                for kernel_name in sorted_kernel_names:
                    value = kernel_summaries[kernel_name].get(metric, 0)
                    row.append(value)
                rows.append(row)
        else:
            # Update header
            rows[0].extend(column_names)
            
            # Update or add metric rows
            metric_row_map = {rows[i][0]: i for i in range(1, len(rows)) if rows[i] and rows[i][0]}
            
            for metric in ordered_metrics:
                if metric in metric_row_map:
                    # Update existing row
                    row_idx = metric_row_map[metric]
                    for kernel_name in sorted_kernel_names:
                        value = kernel_summaries[kernel_name].get(metric, 0)
                        rows[row_idx].append(value)
                else:
                    # Add new row with zeros for previous columns
                    new_row = [metric] + ['0'] * (len(rows[0]) - len(column_names) - 1)
                    for kernel_name in sorted_kernel_names:
                        value = kernel_summaries[kernel_name].get(metric, 0)
                        new_row.append(value)
                    rows.append(new_row)
        
        # Write updated data back
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as writefile:
            writer = csv.writer(writefile)
            writer.writerows(rows)
    
    # Print summary
    print(f"Successfully processed {len(individual_data)} kernels from {log_file_name}")
    for kernel_name in sorted_kernel_names:
        print(f"  - {kernel_name}: {kernel_counts[kernel_name]} instances summed")

def process_filelist(filelist_path, output_csv_path):
    """
    Process a list of log files and aggregate results into a single CSV.
    
    Args:
        filelist_path (str): Path to file containing list of log files
        output_csv_path (str): Path to output CSV file
    """
    try:
        with open(filelist_path, 'r') as f:
            log_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Filelist '{filelist_path}' not found.")
        return
    
    print(f"Processing {len(log_files)} log files...")
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"Warning: Log file '{log_file}' not found, skipping...")
            continue
        
        print(f"\nProcessing: {log_file}")
        kernels_data, kernel_names = parse_occupancy_distributions(log_file)
        
        if kernels_data:
            write_to_csv(output_csv_path, os.path.basename(log_file), kernels_data, kernel_names)
        else:
            print(f"No data found in {log_file}")

def main():
    """
    Main function to run the parser.
    Takes command line arguments for input filelist and output CSV.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse multiple log files and extract occupancy distributions to CSV")
    parser.add_argument("filelist", help="Path to file containing list of log files to process")
    parser.add_argument("-o", "--output", default="occupancy_distributions.csv", 
                       help="Path to the output CSV file (default: occupancy_distributions.csv)")
    
    args = parser.parse_args()
    
    process_filelist(args.filelist, args.output)

if __name__ == "__main__":
    main()