import SR_2
import numpy as np

# Setup Material
mat = SR_2.Material('Eco', 50e6, 125000, {}, 'Neo-Hookean')

# Setup Robots
segs_100 = [SR_2.Segment(0.04, 0.008, 0.0012, 0.0008, mat) for _ in range(100)]
robot_100 = SR_2.SoftRobotModel(segs_100)

segs_5 = [SR_2.Segment(0.04, 0.008, 0.0012, 0.0008, mat) for _ in range(5)]
robot_5 = SR_2.SoftRobotModel(segs_5)

target = np.array([0.108, 0, 0.167])

print(f"Target: {target}")

print("\n--- Testing 100 Segments ---")
print(f"Total Rest Length: {sum([s.L for s in segs_100]):.2f}m")
res = robot_100.solve_inverse_kinematics_4channel(target)
print(f"Success: {res['success']}, Error: {res['error']:.4f}m")

print("\n--- Testing 5 Segments ---")
print(f"Total Rest Length: {sum([s.L for s in segs_5]):.2f}m")
res5 = robot_5.solve_inverse_kinematics_4channel(target)
print(f"Success: {res5['success']}, Error: {res5['error']:.4f}m")
