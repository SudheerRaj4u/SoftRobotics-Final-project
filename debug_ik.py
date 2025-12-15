import numpy as np
from SR_2 import Material, Segment, SoftRobotModel, MATERIALS_DB

def test_ik():
    # Setup Robot
    mat_data = MATERIALS_DB['1'] # EcoFlex 0050
    material = Material(mat_data['name'], mat_data['K'], mat_data['mu_initial'], mat_data['params'], 'Neo-Hookean')
    
    # Standard segments
    segments = [
        Segment(0.050, 0.0061, 0.0010, 0.0008, material),
        Segment(0.040, 0.009, 0.0012, 0.0008, material),
        Segment(0.035, 0.008, 0.0011, 0.0008, material),
        Segment(0.050, 0.010, 0.0015, 0.0008, material),
        Segment(0.025, 0.006, 0.0009, 0.0008, material)
    ]
    robot = SoftRobotModel(segments)
    
    # Target Position - Try a simpler target closer to the robot
    # Let's use a point known to be reachable (small bending)
    print("\nTesting with a simple target...")
    simple_target = np.array([0.02, 0.0, 0.15])  # Small X offset, straight up
    print(f"Simple Target: {simple_target}")
    
    # Debug: Check 50kPa for collision
    print("\nChecking 50kPa configuration...")
    _, p_50 = robot.calculate_forward_kinematics(50000, 0, 0, 0)
    points_50 = robot.get_backbone_positions()
    min_z = min([p[2] for p in points_50])
    print(f"Tip at 50kPa: {p_50}")
    print(f"Min Z at 50kPa: {min_z}")
    
    # Run IK
    print("\nRunning IK...")
    result = robot.solve_inverse_kinematics_4channel(simple_target)
    
    print("\n--- IK Result ---")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Pressures: {result['pressures']}")
    print(f"Error: {result['error']}")
    print(f"Iterations: {result['iterations']}")

if __name__ == "__main__":
    test_ik()
