import SR_2
import numpy as np

def find_points():
    print("Finding 10 safe convergence points for 5-segment robot...")
    
    # Setup Robot
    mat = SR_2.Material('Eco', 50e6, 125000, {}, 'Neo-Hookean')
    segs = [SR_2.Segment(0.04, 0.008, 0.0012, 0.0008, mat) for _ in range(10)]
    robot = SR_2.SoftRobotModel(segs)
    
    safe_points = []
    
    # Try random pressures until we get 10 good ones
    count = 0
    attempts = 0
    while count < 10 and attempts < 500:
        attempts += 1
        
        # Random pressures (0 to 40kPa - easier shapes)
        # Using 4 channels
        pressures = np.random.uniform(0, 40000, 4)
        
        # 1. Get Target via FK (Guaranteed reachable physically)
        try:
            _, target_tip = robot.calculate_forward_kinematics(*pressures)
            
            # Check collisions
            pts = robot.get_backbone_positions()
            if min([p[2] for p in pts]) < 0:
                continue # hit ground
                
            # 2. Verify with IK (Does the solver actually find it?)
            ik_res = robot.solve_inverse_kinematics_4channel(target_tip, tolerance=0.002, max_iterations=200)
            
            if ik_res['success']:
                count += 1
                safe_points.append({
                    'id': count,
                    'target': target_tip,
                    'input_pressures': pressures,
                    'solved_pressures': ik_res['pressures'],
                    'error': ik_res['error']
                })
                print(f"  Point {count}: Found valid target at {target_tip}")
        except Exception as e:
            pass

    print("\n--- 10 Safe Converging Points ---")
    print(f"{'ID':<4} {'X (m)':<10} {'Y (m)':<10} {'Z (m)':<10} {'Original P1 (Pa)':<15}")
    print("-" * 60)
    for p in safe_points:
        t = p['target']
        print(f"{p['id']:<4} {t[0]:<10.4f} {t[1]:<10.4f} {t[2]:<10.4f} {p['input_pressures'][0]:<15.0f}")

if __name__ == "__main__":
    find_points()
