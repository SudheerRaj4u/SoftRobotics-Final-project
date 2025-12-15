import SR_2
import numpy as np

# Setup Material
mat = SR_2.Material('Eco', 50e6, 125000, {}, 'Neo-Hookean')

# Setup 2-Segment Robot
# Total Length = 2 * 0.04 = 0.08m
segs_2 = [SR_2.Segment(0.04, 0.008, 0.0012, 0.0008, mat) for _ in range(2)]
robot_2 = SR_2.SoftRobotModel(segs_2)

target = np.array([0.108, 0, 0.167])

print(f"Target: {target}")
print(f"Total Rest Length: {sum([s.L for s in segs_2]):.2f}m")

print("\n--- Running IK for 2 Segments ---")
res = robot_2.solve_inverse_kinematics_4channel(target, tolerance=0.002, max_iterations=200)

print(f"Success: {res['success']}")
print(f"Pressures: {res['pressures']}")
print(f"Final Error: {res['error']*1000:.2f} mm")
print(f"Achieved Pos: {res['tip_position']}")
print(f"Message: {res['message']}")
