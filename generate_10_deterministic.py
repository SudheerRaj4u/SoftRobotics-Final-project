import SR_2
import numpy as np

def generate():
    print("Generating 10 safe points using deterministic inputs...", flush=True)
    mat = SR_2.Material('Eco', 50e6, 125000, {}, 'Neo-Hookean')
    segs = [SR_2.Segment(0.04, 0.008, 0.0012, 0.0008, mat) for _ in range(10)]
    robot = SR_2.SoftRobotModel(segs)
    
    # List of "easy" pressure combinations likely to converge
    # For 10 segments, keep pressures low to avoid crazy coiling
    test_pressures = [
        [5000, 0, 0, 0],
        [10000, 0, 0, 0],
        [15000, 0, 0, 0],
        [0, 5000, 0, 0],
        [0, 10000, 0, 0],
        [5000, 5000, 0, 0], # Diagonal
        [10000, 10000, 0, 0],
        [8000, 0, 8000, 0], # Agonist-Antagonist (should be straight? No, P1/P3. Net moment 0) -> Tip [0,0,Z]
        [0, 0, 5000, 0],
        [0, 0, 0, 5000],
        [2000, 4000, 0, 0],
        [5000, 0, 0, 5000] # Twist? No, planar 4-channel. P1(0), P4(270).
    ]
    
    safe_points = []
    
    for i, p_in in enumerate(test_pressures):
        if len(safe_points) >= 10: break
        
        print(f"Testing config {i+1}: {p_in}...", end='', flush=True)
        try:
            # FK
            _, target = robot.calculate_forward_kinematics(*p_in)
            
            # IK
            res = robot.solve_inverse_kinematics_4channel(target, max_iterations=200, tolerance=0.002)
            
            if res['success']:
                print(" SUCCESS", flush=True)
                safe_points.append({
                    'id': len(safe_points)+1,
                    'target': target,
                    'p_in': p_in
                })
            else:
                print(" FAIL", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            
    print("\n--- 10 Safe Convergence Points (10 Segments) ---")
    print(f"{'ID':<4} {'X (m)':<10} {'Y (m)':<10} {'Z (m)':<10} {'Input Pressures (Pa)':<30}")
    print("-" * 80)
    for p in safe_points:
        t = p['target']
        press = str(p['p_in'])
        print(f"{p['id']:<4} {t[0]:<10.4f} {t[1]:<10.4f} {t[2]:<10.4f} {press:<30}")

if __name__ == "__main__":
    generate()
