import matplotlib.pyplot as plt
import re
from collections import defaultdict
import sys
import matplotlib.colors as mcolors

def parse_log_file(file_path):
    block_instances = []
    
    patterns = {
        'FETCH_META': r'\[FETCH_(META_START|META_END)\]: Cycle (\d+), hw_cta=(\d+), Block=(\d+)',
        'FETCH_BITS': r'\[FETCH_(BITS_START|BITS_END)\]: Cycle (\d+), hw_cta=(\d+), Block=(\d+)',
        'DP_CGRA': r'\[(DISPATCH_START|CGRA_EXECU_END)\]: Cycle (\d+), hw_cta=(\d+), Block=(\d+)',
        'MEM_WRITEBACK': r'\[WRITEBACK_(START|END)\]: Cycle (\d+), hw_cta=(\d+), Block=(\d+), table_index=(\d+)'
    }
    
    temp_stages = defaultdict(lambda: defaultdict(list))
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                for stage, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        event_type = match.group(1)
                        cycle = int(match.group(2))
                        hw_cta = int(match.group(3))
                        block = int(match.group(4))
                        table_index = int(match.group(5)) if stage == 'MEM_WRITEBACK' else None
                        
                        key = (hw_cta, block, table_index) if stage == 'MEM_WRITEBACK' else (hw_cta, block)
                        
                        if 'START' in event_type or event_type == 'DISPATCH_START':
                            temp_stages[stage][key].append({'start': cycle})
                        elif 'END' in event_type or event_type == 'CGRA_EXECU_END':
                            for instance in reversed(temp_stages[stage][key]):
                                if 'end' not in instance:
                                    instance['end'] = cycle
                                    block_instances.append({
                                        'stage': stage,
                                        'hw_cta': hw_cta,
                                        'block': block,
                                        'start': instance['start'],
                                        'end': cycle,
                                        'table_index': table_index if stage == 'MEM_WRITEBACK' else None
                                    })
                                    break
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        sys.exit(1)
    
    return block_instances

def create_timeline_chart(block_instances, output_file='timeline.png'):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    base_positions = {
        'FETCH_META': 4,
        'FETCH_BITS': 3,
        'DP_CGRA': 2,
        'MEM_WRITEBACK': 1
    }
    
    all_pairs = set((instance['hw_cta'], instance['block']) for instance in block_instances)
    colors = plt.cm.get_cmap('tab10', len(all_pairs) if len(all_pairs) <= 10 else 10)
    pair_colors = {pair: colors(i % 10) for i, pair in enumerate(sorted(all_pairs))}
    
    y_positions = {}
    mem_wb_rows = defaultdict(list)
    
    # Assign positions for non-MEM_WRITEBACK stages
    for instance in block_instances:
        if instance['stage'] != 'MEM_WRITEBACK':
            y_positions[(instance['stage'], instance['hw_cta'], instance['block'])] = base_positions[instance['stage']]
    
    # Collect and sort MEM_WRITEBACK instances
    mem_wb_instances = sorted(
        [i for i in block_instances if i['stage'] == 'MEM_WRITEBACK'],
        key=lambda x: (x['start'], x['end'])
    )
    
    # Place MEM_WRITEBACK instances based on table_index
    for instance in mem_wb_instances:
        start = instance['start']
        end = instance['end']
        hw_cta = instance['hw_cta']
        block = instance['block']
        table_index = instance['table_index']
        
        key = ('MEM_WRITEBACK', hw_cta, block, table_index)
        mem_wb_rows[table_index].append((start, end, hw_cta, block))
        y_positions[key] = base_positions['MEM_WRITEBACK'] - table_index * 0.8
    
    # Debug y_positions for MEM_WRITEBACK
    print("Y-Positions for MEM_WRITEBACK:")
    for key, pos in y_positions.items():
        if key[0] == 'MEM_WRITEBACK':
            print(f"Key: {key}, Y-Position: {pos}")
    
    # Plot all instances
    for instance in block_instances:
        stage = instance['stage']
        hw_cta = instance['hw_cta']
        block = instance['block']
        start = instance['start']
        end = instance['end']
        
        if stage == 'MEM_WRITEBACK':
            key = (stage, hw_cta, block, instance['table_index'])
        else:
            key = (stage, hw_cta, block)
        
        y_pos = y_positions[key]
        duration = end - start
        
        rect = plt.Rectangle((start, y_pos-0.4), duration, 0.8,
                            facecolor=pair_colors[(hw_cta, block)],
                            alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        # Change legend to vertical format: (hw_cta)\n(block)
        ax.text(start + duration/2, y_pos, f'({hw_cta})\n({block})',
                ha='center', va='center', color='black', fontsize=8)
    
    # Customize plot
    yticks = []
    yticklabels = []
    for stage in base_positions:
        if stage != 'MEM_WRITEBACK':
            yticks.append(base_positions[stage])
            yticklabels.append(stage)
        else:
            max_table_index = max(mem_wb_rows.keys()) if mem_wb_rows else 0
            for row_idx in range(max_table_index + 1):
                yticks.append(base_positions[stage] - row_idx * 0.8)
                yticklabels.append(f'MEM_WRITEBACK Lane {row_idx}')
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Cycles')
    ax.set_title('Hardware Execution Stages Timeline')
    
    all_cycles = [instance['start'] for instance in block_instances] + \
                 [instance['end'] for instance in block_instances]
    if all_cycles:
        ax.set_xlim(min(all_cycles)-10, max(all_cycles)+10)
    
    ax.set_ylim(min(yticks)-0.5, max(yticks)+0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Debug output for lanes
    for table_idx in sorted(mem_wb_rows.keys()):
        print(f"Lane {table_idx}: {sorted(mem_wb_rows[table_idx], key=lambda x: x[0])}")
    
    plt.savefig(output_file)
    plt.close()
    print(f"Timeline chart saved as {output_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 create_timeline.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = log_file.rsplit('.', 1)[0] + '_timeline.png'
    
    block_instances = parse_log_file(log_file)
    create_timeline_chart(block_instances, output_file)

if __name__ == "__main__":
    main()