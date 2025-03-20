import matplotlib.pyplot as plt
import re
from collections import defaultdict
import sys
import matplotlib.colors as mcolors

def parse_log_file(file_path):
    stages = defaultdict(lambda: defaultdict(dict))
    
    patterns = {
        'FETCH_META': r'\[FETCH_(META_START|META_END)\]: Cycle (\d+), Block=(\d+)',
        'FETCH_BITS': r'\[FETCH_(BITS_START|BITS_END)\]: Cycle (\d+), Block=(\d+)',
        'DP_CGRA': r'\[(DISPATCH_START|CGRA_EXECU_END)\]: Cycle (\d+), Block=(\d+)',
        'MEM_WRITEBACK': r'\[WRITEBACK_(START|END)\]: Cycle (\d+), Block=(\d+)'
    }
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                for stage, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        event_type = match.group(1)
                        cycle = int(match.group(2))
                        block = int(match.group(3))
                        
                        if 'START' in event_type or event_type == 'DISPATCH_START':
                            stages[stage][block]['start'] = cycle
                        elif 'END' in event_type or event_type == 'CGRA_EXECU_END':
                            stages[stage][block]['end'] = cycle
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        sys.exit(1)
    
    return stages

def create_timeline_chart(stages, output_file='timeline.png'):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stage_positions = {
        'FETCH_META': 4,
        'FETCH_BITS': 3,
        'DP_CGRA': 2,
        'MEM_WRITEBACK': 1
    }
    
    # Get all unique block IDs
    all_blocks = set()
    for stage in stages.values():
        all_blocks.update(stage.keys())
    
    # Create a color map for blocks
    colors = plt.cm.get_cmap('tab10', len(all_blocks) if len(all_blocks) <= 10 else 10)
    block_colors = {block: colors(i % 10) for i, block in enumerate(sorted(all_blocks))}
    
    # Plot each stage
    for stage, blocks in stages.items():
        y_pos = stage_positions[stage]
        for block, times in blocks.items():
            if 'start' in times and 'end' in times:
                start = times['start']
                duration = times['end'] - start
                # Plot rectangle with block-specific color
                rect = plt.Rectangle((start, y_pos-0.4), duration, 0.8,
                                   facecolor=block_colors[block],
                                   alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
                # Add block number at the top with black color
                ax.text(start + duration/2, y_pos + 0.4, f'{block}',
                       ha='center', va='bottom', color='black')  # Fixed to black
    
    # Customize the plot
    ax.set_yticks(list(stage_positions.values()))
    ax.set_yticklabels(list(stage_positions.keys()))
    ax.set_xlabel('Cycles')
    ax.set_title('Hardware Execution Stages Timeline')
    
    # Set x-axis limits based on data
    all_cycles = []
    for stage in stages.values():
        for times in stage.values():
            if 'start' in times:
                all_cycles.append(times['start'])
            if 'end' in times:
                all_cycles.append(times['end'])
    if all_cycles:
        ax.set_xlim(min(all_cycles)-10, max(all_cycles)+10)
    
    ax.set_ylim(0.5, 4.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()
    print(f"Timeline chart saved as {output_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 create_timeline.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = log_file.rsplit('.', 1)[0] + '_timeline.png'
    
    stages = parse_log_file(log_file)
    create_timeline_chart(stages, output_file)

if __name__ == "__main__":
    main()