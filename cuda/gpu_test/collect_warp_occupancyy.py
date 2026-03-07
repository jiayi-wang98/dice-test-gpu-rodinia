import re
import csv
import os
from collections import defaultdict

def parse_filelist(filelist_path, output_csv_path):
    """
    Parse multiple kernel log files from a filelist and extract warp occupancy data into CSV format.
    
    Args:
        filelist_path (str): Path to the filelist containing log file paths
        output_csv_path (str): Path to the output CSV file
    """
    
    # Dictionary to store aggregated data by kernel name
    aggregated_kernel_data = defaultdict(lambda: defaultdict(int))
    # List to maintain kernel order as they appear
    kernel_order = []
    
    try:
        # Read the filelist
        with open(filelist_path, 'r') as f:
            log_files = [line.strip() for line in f if line.strip()]
        
        if not log_files:
            print(f"No files found in filelist: {filelist_path}")
            return
        
        print(f"Processing {len(log_files)} log files...")
        
        # Process each log file
        for log_file in log_files:
            if not os.path.exists(log_file):
                print(f"Warning: File '{log_file}' not found, skipping...")
                continue
            
            print(f"Processing: {log_file}")
            kernel_data_by_name, kernel_order_in_file = parse_single_kernel_log(log_file)
            
            # Aggregate data from this file
            for kernel_name, metrics in kernel_data_by_name.items():
                # Track kernel order
                if kernel_name not in kernel_order:
                    kernel_order.append(kernel_name)
                    
                for metric, value in metrics.items():
                    aggregated_kernel_data[kernel_name][metric] += value
    
    except FileNotFoundError:
        print(f"Error: Filelist '{filelist_path}' not found.")
        return
    except Exception as e:
        print(f"Error reading filelist: {e}")
        return
    
    if not aggregated_kernel_data:
        print("No kernel data found in any of the files.")
        return
    
    # Write to CSV
    write_to_csv(aggregated_kernel_data, kernel_order, output_csv_path)
    print(f"Successfully parsed and aggregated data for {len(aggregated_kernel_data)} unique kernels from {len(log_files)} files")

def parse_single_kernel_log(log_file_path):
    """
    Parse a single kernel log file and extract warp occupancy data.
    Returns aggregated data by kernel name and the order of first appearance.
    
    Args:
        log_file_path (str): Path to the input log file
        
    Returns:
        tuple: (aggregated_data_dict, kernel_order_list)
    """
    
    # Dictionary to store cumulative data and names for each kernel launch
    kernels_cumulative_data = {}
    kernel_names = {}
    kernel_first_appearance = []  # Track order of first appearance
    
    # Regular expressions for parsing
    kernel_name_pattern = r'kernel_name\s*=\s*(.+?)(?=\n|\r|$)'
    kernel_uid_pattern = r'kernel_launch_uid\s*=\s*(\d+)'
    warp_occupancy_pattern = r'Warp Occupancy Distribution:\s*(.+)'
    
    try:
        with open(log_file_path, 'r') as file:
            content = file.read()
            
        # Find all kernel names, UIDs and their corresponding warp occupancy data
        kernel_name_matches = list(re.finditer(kernel_name_pattern, content))
        kernel_uid_matches = list(re.finditer(kernel_uid_pattern, content))
        warp_occupancy_matches = list(re.finditer(warp_occupancy_pattern, content))
        
        # Convert to lists with positions for easier handling
        name_positions = [(match.group(1).strip(), match.start()) for match in kernel_name_matches]
        uid_positions = [(match.group(1), match.start()) for match in kernel_uid_matches]
        warp_positions = [(match.group(1), match.start()) for match in warp_occupancy_matches]
        
        # Match each warp occupancy with its corresponding kernel UID and name
        for warp_data, warp_pos in warp_positions:
            # Find the most recent kernel UID before this warp occupancy data
            matching_uid = None
            matching_name = None
            
            # Find matching UID
            for uid, uid_pos in reversed(uid_positions):
                if uid_pos < warp_pos:
                    matching_uid = uid
                    break
            
            # Find matching name
            for name, name_pos in reversed(name_positions):
                if name_pos < warp_pos:
                    matching_name = name
                    break
            
            if matching_uid:
                # Parse the warp occupancy data
                parsed_data = parse_warp_occupancy(warp_data)
                kernels_cumulative_data[matching_uid] = parsed_data
                kernel_names[matching_uid] = matching_name or f"Unknown_Kernel_{matching_uid}"
                
                # Track first appearance order
                kernel_name = matching_name or f"Unknown_Kernel_{matching_uid}"
                if kernel_name not in kernel_first_appearance:
                    kernel_first_appearance.append(kernel_name)
    
    except Exception as e:
        print(f"Error processing file {log_file_path}: {e}")
        return {}, []
    
    # Convert cumulative data to individual kernel launches
    individual_data = calculate_individual_kernel_data(kernels_cumulative_data)
    
    # Aggregate by kernel name
    aggregated_by_name = defaultdict(lambda: defaultdict(int))
    for uid, metrics in individual_data.items():
        kernel_name = kernel_names.get(uid, f"Unknown_Kernel_{uid}")
        for metric, value in metrics.items():
            aggregated_by_name[kernel_name][metric] += value
    
    return dict(aggregated_by_name), kernel_first_appearance

def parse_warp_occupancy(warp_data):
    """
    Parse warp occupancy string and extract metric names and values.
    Keep original metric names without grouping.
    
    Args:
        warp_data (str): Raw warp occupancy string
        
    Returns:
        dict: Dictionary with original metric names as keys and values as integers
    """
    parsed = {}
    
    # Split by tabs and parse each metric:value pair
    parts = warp_data.split('\t')
    
    for part in parts:
        part = part.strip()
        if ':' in part:
            metric, value = part.split(':', 1)
            try:
                parsed[metric.strip()] = int(value.strip())
            except ValueError:
                print(f"Warning: Could not parse value '{value}' for metric '{metric}'")
                parsed[metric.strip()] = 0
    
    return parsed

def calculate_individual_kernel_data(kernels_cumulative_data):
    """
    Convert cumulative kernel data to individual kernel data by subtracting previous kernels.
    
    Args:
        kernels_cumulative_data (dict): Dictionary with kernel UIDs as keys and cumulative data as values
        
    Returns:
        dict: Dictionary with individual kernel data
    """
    # Sort kernel UIDs numerically to process in order
    sorted_kernel_uids = sorted(kernels_cumulative_data.keys(), key=int)
    
    # Get all unique metric names
    all_metrics = set()
    for data in kernels_cumulative_data.values():
        all_metrics.update(data.keys())
    
    individual_data = {}
    previous_cumulative = defaultdict(int)
    
    for uid in sorted_kernel_uids:
        individual_data[uid] = {}
        current_cumulative = kernels_cumulative_data[uid]
        
        for metric in all_metrics:
            current_value = current_cumulative.get(metric, 0)
            previous_value = previous_cumulative[metric]
            
            # Individual kernel value = current cumulative - previous cumulative
            individual_value = current_value - previous_value
            individual_data[uid][metric] = individual_value
        
        # Update previous cumulative for next iteration
        previous_cumulative = defaultdict(int, current_cumulative)
    
    return individual_data

def write_to_csv(aggregated_kernel_data, kernel_order, output_csv_path):
    """
    Write the aggregated kernel data to CSV file.
    
    Args:
        aggregated_kernel_data (dict): Dictionary with kernel names as keys and aggregated metric data as values
        kernel_order (list): List of kernel names in order of first appearance
        output_csv_path (str): Path to output CSV file
    """
    if not aggregated_kernel_data:
        print("No kernel data found to write.")
        return
    
    # Get all unique metrics
    all_metrics = set()
    for metrics in aggregated_kernel_data.values():
        all_metrics.update(metrics.keys())
    
    # Sort metrics naturally (handle W0, W1, W10, W11 ordering correctly)
    def metric_sort_key(metric):
        # Extract number from W metrics
        if metric.startswith('W') and metric[1:].isdigit():
            return (0, int(metric[1:]))
        elif metric.startswith('W') and '_' in metric:
            # Handle W0_Idle, W0_Scoreboard
            parts = metric.split('_')
            if parts[0][1:].isdigit():
                return (1, int(parts[0][1:]), parts[1])
        # Put Stall at the beginning
        elif metric == 'Stall':
            return (-1, 0)
        # Other metrics at the end
        return (2, metric)
    
    sorted_metrics = sorted(all_metrics, key=metric_sort_key)
    
    # Use the kernel order from file appearance instead of sorting
    ordered_kernel_names = kernel_order
    
    # Calculate total sum across all kernels
    total_sum = defaultdict(int)
    for kernel_metrics in aggregated_kernel_data.values():
        for metric, value in kernel_metrics.items():
            total_sum[metric] += value
    
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header row
            header = ['Metric'] + ordered_kernel_names + ['Total_Sum']
            writer.writerow(header)
            
            # Write data rows in sorted order
            for metric in sorted_metrics:
                row = [metric]
                
                # Kernel values
                for kernel_name in ordered_kernel_names:
                    value = aggregated_kernel_data[kernel_name].get(metric, 0)
                    row.append(value)
                
                # Total sum
                row.append(total_sum[metric])
                
                writer.writerow(row)
                
    except Exception as e:
        print(f"Error writing CSV file: {e}")

def main():
    """
    Main function to run the parser.
    Takes command line arguments for input filelist and output file.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse multiple kernel log files from a filelist and extract warp occupancy data to CSV")
    parser.add_argument("filelist", help="Path to the filelist containing log file paths")
    parser.add_argument("-o", "--output", default="kernel_occupancy.csv", 
                       help="Path to the output CSV file (default: kernel_occupancy.csv)")
    
    args = parser.parse_args()
    
    parse_filelist(args.filelist, args.output)

if __name__ == "__main__":
    main()