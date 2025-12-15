import SR_2
import numpy as np

# Setup Material
mat = SR_2.Material('Eco', 50e6, 125000, {}, 'Neo-Hookean')

# Setup 6-Segment Robot
# Total Length = 6 * 0.04 = 0.24m
segs_6 = [SR_2.Segment(0.04, 0.008, 0.0012, 0.0008, mat) for _ in range(6)]
robot_6 = SR_2.SoftRobotModel(segs_6)

target = np.array([0.108, 0, 0.167])

print(f"Target: {target}")
print(f"Total Rest Length: {sum([s.L for s in segs_6]):.2f}m")

print("\n--- Running IK for 6 Segments ---")
# Using same params as SR_2.py main: tolerance=0.002, max_iterations=200
res = robot_6.solve_inverse_kinematics_4channel(target, tolerance=0.002, max_iterations=200)

print(f"Success: {res['success']}")
print(f"Pressures: {res['pressures']}")
print(f"Final Error: {res['error']*1000:.2f} mm")
print(f"Achieved Pos: {res['tip_position']}")
print(f"Message: {res['message']}")
