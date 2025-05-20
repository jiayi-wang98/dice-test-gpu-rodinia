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

def extract_stats(file_path, output_file):
    shader_stats = {}
    global_stats = {}
    
    patterns = {
        'gpu_total_sim_cycle': r'gpu_tot_sim_cycle = (\d+)',
        'shader_start': r'SHADER (\d+):',
        'read_regfile': r'gpgpu_n_m_read_regfile_acesses = (\d+)',
        'write_regfile': r'gpgpu_n_m_write_regfile_acesses = (\d+)',
        'tot_regfile': r'gpgpu_n_tot_regfile_acesses = (\d+)',
        'l1i_accesses': r'L1I_total_cache_accesses = (\d+)',
        'l1b_accesses': r'L1B_total_cache_accesses = (\d+)',
        'l1c_accesses': r'L1C_total_cache_accesses = (\d+)',
        'l1t_accesses': r'L1T_total_cache_accesses = (\d+)',
        'l1d_accesses': r'L1D_total_cache_accesses = (\d+)',
        'l1d_misses': r'L1D_total_cache_misses = (\d+)',
        'l1d_miss_rate': r'L1D_total_cache_miss_rate = ([\d.]+)',
        'l2_accesses': r'L2_total_cache_accesses = (\d+)',
        'l2_misses': r'L2_total_cache_misses = (\d+)',
        'l2_miss_rate': r'L2_total_cache_miss_rate = ([\d.]+)'
    }
    
    current_shader = None
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Check for shader start
                match = re.search(patterns['shader_start'], line)
                if match:
                    current_shader = int(match.group(1))
                    shader_stats[current_shader] = {}
                    continue
                
                # Shader-specific stats
                if current_shader is not None:
                    read_match = re.search(patterns['read_regfile'], line)
                    if read_match:
                        shader_stats[current_shader]['read_regfile'] = int(read_match.group(1))
                    
                    write_match = re.search(patterns['write_regfile'], line)
                    if write_match:
                        shader_stats[current_shader]['write_regfile'] = int(write_match.group(1))
                        current_shader = None  # Reset after last stat for this shader
                
                # Global stats
                for key, pattern in patterns.items():
                    if key in ['shader_start', 'read_regfile', 'write_regfile']:
                        continue
                    match = re.search(pattern, line)
                    if match:
                        value = int(match.group(1)) if key in ['tot_regfile', 'l1d_accesses', 'l1d_misses', 'l2_accesses', 'l2_misses'] else float(match.group(1))
                        global_stats[key] = value
    
        # Write stats to output file
        with open(output_file, 'w') as out_file:
            out_file.write("Shader Statistics:\n")
            out_file.write(f"gpu_tot_sim_cycle = {global_stats.get('gpu_total_sim_cycle', 0)}\n")
            for shader_id, stats in sorted(shader_stats.items()):
                out_file.write(f"SHADER {shader_id}:\n")
                out_file.write(f"  gpgpu_n_m_read_regfile_acesses = {stats.get('read_regfile', 0)}\n")
                out_file.write(f"  gpgpu_n_m_write_regfile_acesses = {stats.get('write_regfile', 0)}\n")
            out_file.write("\nGlobal Statistics:\n")
            out_file.write(f"gpgpu_n_tot_regfile_acesses = {global_stats.get('tot_regfile', 0)}\n")
            out_file.write(f"L1I_total_cache_accesses = {global_stats.get('l1i_accesses', 0)}\n")
            out_file.write(f"L1B_total_cache_accesses = {global_stats.get('l1b_accesses', 0)}\n")
            out_file.write(f"L1C_total_cache_accesses = {global_stats.get('l1c_accesses', 0)}\n")
            out_file.write(f"L1T_total_cache_accesses = {global_stats.get('l1t_accesses', 0)}\n")
            out_file.write(f"L1D_total_cache_accesses = {global_stats.get('l1d_accesses', 0)}\n")
            out_file.write(f"L1D_total_cache_misses = {global_stats.get('l1d_misses', 0)}\n")
            out_file.write(f"L1D_total_cache_miss_rate = {global_stats.get('l1d_miss_rate', 0.0)}\n")

            # Assuming global_stats is a dictionary containing the cache access counts
            total_l1_cache_accesses = (
                global_stats.get('l1i_accesses', 0) +
                global_stats.get('l1b_accesses', 0) +
                global_stats.get('l1c_accesses', 0) +
                global_stats.get('l1t_accesses', 0) +
                global_stats.get('l1d_accesses', 0)
            )

            # Optionally, write to the output file
            out_file.write(f"Total_L1_cache_accesses = {total_l1_cache_accesses}\n")

            out_file.write(f"L2_total_cache_accesses = {global_stats.get('l2_accesses', 0)}\n")
            out_file.write(f"L2_total_cache_misses = {global_stats.get('l2_misses', 0)}\n")
            out_file.write(f"L2_total_cache_miss_rate = {global_stats.get('l2_miss_rate', 0.0)}\n")
        
        print(f"Statistics written to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing stats: {str(e)}")
        sys.exit(1)

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
    
    for instance in block_instances:
        if instance['stage'] != 'MEM_WRITEBACK':
            y_positions[(instance['stage'], instance['hw_cta'], instance['block'])] = base_positions[instance['stage']]
    
    mem_wb_instances = sorted(
        [i for i in block_instances if i['stage'] == 'MEM_WRITEBACK'],
        key=lambda x: (x['start'], x['end'])
    )
    
    for instance in mem_wb_instances:
        start = instance['start']
        end = instance['end']
        hw_cta = instance['hw_cta']
        block = instance['block']
        table_index = instance['table_index']
        
        key = ('MEM_WRITEBACK', hw_cta, block, table_index)
        mem_wb_rows[table_index].append((start, end, hw_cta, block))
        y_positions[key] = base_positions['MEM_WRITEBACK'] - table_index * 0.8
    
    print("Y-Positions for MEM_WRITEBACK:")
    for key, pos in y_positions.items():
        if key[0] == 'MEM_WRITEBACK':
            print(f"Key: {key}, Y-Position: {pos}")
    
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
        ax.text(start + duration/2, y_pos, f'({hw_cta})\n({block})',
                ha='center', va='center', color='black', fontsize=8)
    
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
    stats_file = log_file.rsplit('.', 1)[0] + '_status.info'
    
    block_instances = parse_log_file(log_file)
    create_timeline_chart(block_instances, output_file)
    extract_stats(log_file, stats_file)

if __name__ == "__main__":
    main()