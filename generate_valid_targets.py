import SR_2
import numpy as np

def get_reachable_point(num_segments, pressures):
    mat = SR_2.Material('Eco', 50e6, 125000, {}, 'Neo-Hookean')
    segs = [SR_2.Segment(0.04, 0.008, 0.0012, 0.0008, mat) for _ in range(num_segments)]
    robot = SR_2.SoftRobotModel(segs)
    
    # Run FK
    _, p_tip = robot.calculate_forward_kinematics(*pressures)
    return p_tip

print("--- Valid Reachable Targets ---")

# 2 Segments (Short)
p2 = get_reachable_point(2, [20000, 0, 0, 0])
print(f"\n2 Segments (Length 0.08m):")
print(f"  Input Pressure: P1=20kPa")
print(f"  Target Point: X={p2[0]:.4f}, Y={p2[1]:.4f}, Z={p2[2]:.4f}")

# 6 Segments (Medium)
p6 = get_reachable_point(6, [10000, 10000, 0, 0])
print(f"\n6 Segments (Length 0.24m):")
print(f"  Input Pressure: P1=10kPa, P2=10kPa")
print(f"  Target Point: X={p6[0]:.4f}, Y={p6[1]:.4f}, Z={p6[2]:.4f}")

# 100 Segments (Long)
# Low pressure because it bends easily
p100 = get_reachable_point(100, [500, 0, 0, 0]) 
print(f"\n100 Segments (Length 4.00m):")
print(f"  Input Pressure: P1=500Pa")
print(f"  Target Point: X={p100[0]:.4f}, Y={p100[1]:.4f}, Z={p100[2]:.4f}")
