def filter_log_lines(input_file, output_file):
    keyword = "MEMORY_PARTITION_UNIT -  0"
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if keyword in line:
                outfile.write(line)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python filter_log.py <logfile>")
    else:
        input_path = sys.argv[1]
        output_path = "filtered_output.log"
        filter_log_lines(input_path, output_path)
        print(f"Filtered lines saved to {output_path}")