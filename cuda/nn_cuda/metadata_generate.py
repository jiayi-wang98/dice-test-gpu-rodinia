import re
import sys

def analyze_ptx_block(lines, block_id, block_label, all_blocks):
    analysis = {
        "DBB_ID": block_id,
        "BITSTREAM_ADDR": block_label,
        "BITSTREAM_LENGTH": 0,
        "UNROLLING_FACTOR": 1,
        "UNROLLING_STRATEGY": 0,
        "LAT": 0,
        "IN_REGS": set(),
        "OUT_REGS": set(),
        "LD_DEST_REGS": set(),
        "STORE": 0,
        "BRANCH": 0,
        "BRANCH_UNI": 0,
        "BRANCH_PRED": set(),
        "BRANCH_TARGET": None,
        "BRANCH_RECVPC": None
    }
    
    defined_regs = set()
    used_regs = set()
    instruction_count = 0
    instructions = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('}'):
            continue
            
        if not any(cmd in line.lower() for cmd in ['bra', 'ld', 'st']):
            instruction_count += 1
        instructions.append(line)
        
        regs = re.findall(r'%\w+\d+', line)
        
        if 'ld' in line.lower():
            dest_reg = regs[0] if regs else None
            if dest_reg:
                analysis["LD_DEST_REGS"].add(dest_reg)
                defined_regs.add(dest_reg)
                for reg in regs[1:]:
                    used_regs.add(reg)
                    
        elif 'st' in line.lower():
            analysis["STORE"] += 1
            for reg in regs:
                used_regs.add(reg)
                
        elif 'bra' in line.lower():
            analysis["BRANCH"] = 1
            parts = line.split()
            if parts[0].startswith('@'):
                pred_reg = parts[0][1:].strip('!')
                analysis["BRANCH_PRED"].add(pred_reg)
                used_regs.add(pred_reg)
                target = parts[2].strip(';')
            else:
                analysis["BRANCH_UNI"] = 1
                target = parts[1].strip(';')
            try:
                analysis["BRANCH_TARGET"] = int(target.split('_')[-1])
            except (IndexError, ValueError):
                analysis["BRANCH_TARGET"] = None
                
        else:
            if regs:
                defined_regs.add(regs[0])
                for reg in regs[1:]:
                    used_regs.add(reg)
    
    analysis["BITSTREAM_LENGTH"] = instruction_count
    analysis["LAT"] = len(instructions) // 2
    
    prev_defined = set()
    for i in range(block_id):
        prev_lines = all_blocks[i][1]
        for line in prev_lines:
            regs = re.findall(r'%\w+\d+', line)
            if regs and ('ld' in line.lower() or not any(cmd in line.lower() for cmd in ['st', 'bra'])):
                prev_defined.add(regs[0])
    
    later_used = set()
    for i in range(block_id + 1, len(all_blocks)):
        later_lines = all_blocks[i][1]
        for line in later_lines:
            regs = re.findall(r'%\w+\d+', line)
            for reg in regs:
                later_used.add(reg)
    
    analysis["IN_REGS"] = used_regs & prev_defined
    analysis["OUT_REGS"] = (defined_regs & later_used) - analysis["LD_DEST_REGS"]
    
    return analysis

def analyze_ptx_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    
    ptx_content = ''.join(lines)
    func_match = re.search(r'\.entry\s+([^\(]+)', ptx_content)
    func_name = func_match.group(1) if func_match else "Unknown"
    
    blocks = []
    current_block = []
    current_label = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('$DICE_BB_'):
            if current_block and current_label:
                blocks.append((current_label, current_block))
            current_label = line
            current_block = []
        elif line and not line.startswith('.'):
            current_block.append(line)
    
    if current_block and current_label:
        blocks.append((current_label, current_block))
    
    results = [f"FUNCTION = {func_name};"]
    
    for block_id, (label, block_lines) in enumerate(blocks):
        is_return_block = any('ret' in line.lower() for line in block_lines)
        
        if is_return_block:
            results.append(f"DBB_ID = {block_id},")
            results.append(f"BITSTREAM_ADDR = {label},")
            results.append("RET;")
            continue
            
        analysis = analyze_ptx_block(block_lines, block_id, label, blocks)
        
        # Set IS_PARAMETER_LOAD for the first block (block_id == 0)
        if block_id == 0:
            analysis["IS_PARAMETER_LOAD"] = True
        
        block_output = [f"DBB_ID = {analysis['DBB_ID']},"]
        block_output.append(f"BITSTREAM_ADDR = {analysis['BITSTREAM_ADDR']},")
        block_output.append(f"BITSTREAM_LENGTH = {analysis['BITSTREAM_LENGTH']},")
        block_output.append(f"UNROLLING_FACTOR = {analysis['UNROLLING_FACTOR']},")
        block_output.append(f"UNROLLING_STRATEGY = {analysis['UNROLLING_STRATEGY']},")
        block_output.append(f"LAT = {analysis['LAT']},")
        
        if analysis["IN_REGS"]:
            block_output.append(f"IN_REGS = ({','.join(sorted(analysis['IN_REGS']))}),")
        if analysis["OUT_REGS"]:
            block_output.append(f"OUT_REGS = ({','.join(sorted(analysis['OUT_REGS']))}),")
        if analysis["LD_DEST_REGS"]:
            block_output.append(f"LD_DEST_REGS = ({','.join(sorted(analysis['LD_DEST_REGS']))}),")
            
        block_output.append(f"STORE = {analysis['STORE']},")
        
        if analysis["BRANCH"]:
            block_output.append(f"BRANCH = {analysis['BRANCH']},")
            block_output.append(f"BRANCH_UNI = {analysis['BRANCH_UNI']},")
            if analysis["BRANCH_PRED"]:
                block_output.append(f"BRANCH_PRED = ({','.join(sorted(analysis['BRANCH_PRED']))}),")
            if analysis["BRANCH_TARGET"] is not None:
                block_output.append(f"BRANCH_TARGET = {analysis['BRANCH_TARGET']},")
            block_output.append("BRANCH_RECVPC = 3,")
            
        # Include IS_PARAMETER_LOAD if set (should be for block 0)
        if analysis.get("IS_PARAMETER_LOAD"):
            block_output.append("IS_PARAMETER_LOAD;\n")
        else:
            block_output.append("")
            
        results.extend(block_output)
    
    return '\n'.join(results)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <ptx_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    result = analyze_ptx_file(file_path)
    
    if result:
        # Write to a file instead of printing to console
        output_file = file_path + '.analysis'
        with open(output_file, 'w') as f:
            f.write(result)
        print(f"Analysis saved to: {output_file}")

if __name__ == "__main__":
    main()