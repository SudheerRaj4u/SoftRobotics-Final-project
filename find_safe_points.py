from SR_2 import *
import numpy as np

def find_points():
    # Setup standard robot
    mat = Material("EcoFlex 0050", 50e6, 0.125e6, {'mu_neo': 0.125e6}, "Neo-Hookean")
    segs = [Segment(0.040, 0.009, 0.0012, 0.0008, mat) for _ in range(5)]
    robot = SoftRobotModel(segs)

    print("--- Safe Point 1 (Gentle Bend) ---")
    P1 = [20000, 0, 0, 0] # 20 kPa
    _, tip1 = robot.calculate_forward_kinematics(*P1)
    print(f"Pressures: {P1}")
    print(f"Target: [{tip1[0]:.4f}, {tip1[1]:.4f}, {tip1[2]:.4f}]")

    print("\n--- Safe Point 2 (Diagonal Bend) ---")
    P2 = [30000, 30000, 0, 0] # 30 kPa each
    _, tip2 = robot.calculate_forward_kinematics(*P2)
    print(f"Pressures: {P2}")
    print(f"Target: [{tip2[0]:.4f}, {tip2[1]:.4f}, {tip2[2]:.4f}]")

if __name__ == "__main__":
    find_points()
