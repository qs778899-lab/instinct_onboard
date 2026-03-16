import re

def analyze_full_log(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    frames = content.split('version:')
    if len(frames) < 2:
        return
    
    all_qs = []
    for frame in frames[1:]:
        motor_state_match = re.search(r'motor_state:(.*?)wireless_remote:', frame, re.DOTALL)
        if motor_state_match:
            q_values = re.findall(r'  q: ([-+]?\d*\.\d+|\d+)', motor_state_match.group(1))
            if q_values:
                all_qs.append([float(q) for q in q_values])
    
    if not all_qs:
        print("No data found.")
        return

    num_joints = len(all_qs[0])
    print(f"Analyzing {len(all_qs)} frames for {num_joints} joints.")
    print("Index | Min Q   | Max Q   | Range")
    print("-" * 35)
    for i in range(num_joints):
        qs = [frame[i] for frame in all_qs if i < len(frame)]
        if not qs: continue
        min_q = min(qs)
        max_q = max(qs)
        range_q = max_q - min_q
        if range_q > 0.05:
            print(f"{i:5d} | {min_q:7.3f} | {max_q:7.3f} | {range_q:7.3f} <--- MOVED")
        else:
            print(f"{i:5d} | {min_q:7.3f} | {max_q:7.3f} | {range_q:7.3f}")

analyze_full_log('/home/unitree/yixuan/instinct_onboard/unitree_ros2/lowstate_log.txt')
