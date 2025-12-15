import numpy as np
import sys

# --- 1. Material Model Class ---
class Material:
    """
    This class defines how the "rubber" material behaves.
    Unlike steel, rubber gets stiffer or softer as you stretch it (Hyperelasticity).
    """
    def __init__(self, name, K, mu_initial, params, model_type):
        """
        Initialize the material properties.
        Args:
            name: Human-readable name (e.g., "EcoFlex 0050")
            K: Bulk Modulus (How hard is it to squeeze its volume?)
            mu_initial: Shear Modulus (How hard is it to twist/shear initially?)
            params: Dictionary of extra math constants for the specific model.
            model_type: The mathematical formula to use (Neo-Hookean, etc.)
        """
        self.name = name
        self.K = K                # Bulk Modulus (K) in Pa
        self.mu_initial = mu_initial # Initial Shear Modulus (mu) in Pa
        self.params = params      # Dictionary of model parameters (C10, C01, mu1, alpha1)
        self.model_type = model_type

    def get_tangent_modulus(self, epsilon_pre):
        """
        Calculates the Material Stiffness (Tangential Elastic Modulus) at a specific stretch.
        Think of this as: "How hard is it to stretch *more* given we are already stretched by epsilon_pre?"
        """
        lam = 1 + epsilon_pre # Stretch ratio lambda (e.g., 15% strain -> lambda = 1.15)
        
        if self.model_type == 'Neo-Hookean':
            # Neo-Hookean: The simplest model, good for small-medium stretches.
            # E_tan varies with lambda.
            mu = self.params.get('mu_neo', self.mu_initial)
            E_tan = mu * (2 * lam + lam**(-2))
            return E_tan
            
        elif self.model_type == 'Mooney-Rivlin':
            # Mooney-Rivlin: E_tan = 2 * [ (-C01 * lam^-2)(lam^2 - lam^-1) + (C10 + C01*lam^-1)(2*lam + lam^-2) ]
            C10 = self.params['C10']
            C01 = self.params['C01']
            
            term1 = (-C01 * lam**(-2)) * (lam**2 - lam**(-1))
            term2 = (C10 + C01 * lam**(-1)) * (2 * lam + lam**(-2))
            
            E_tan = 2 * (term1 + term2)
            return E_tan
            
        elif self.model_type == 'Ogden':
            # Ogden (1-term): E_tan = mu1 * [ alpha1 * lam^(alpha1-1) + (alpha1/2) * lam^(-alpha1/2 - 1) ]
            mu1 = self.params['mu1']
            alpha1 = self.params['alpha1']
            
            term1 = alpha1 * lam**(alpha1 - 1)
            term2 = (alpha1 / 2) * lam**(-alpha1/2 - 1)
            
            E_tan = mu1 * (term1 + term2)
            return E_tan
            
        return 3 * self.mu_initial # Fallback

# --- 2. Segment Class ---
class Segment:
    """
    Represents one section (like a vertebrae) of the soft robot arm.
    It holds the dimensions (Length, Radius) and calculates how stiff this specific tube is.
    """
    def __init__(self, L, R_outer, t_w, t_s, material):
        """
        Args:
            L: Length of the segment in meters.
            R_outer: Outer radius of the tube.
            t_w: Wall thickness of the tube.
            t_s: Thickness of the internal septum (divider).
            material: The Material object defined above.
        """
        self.L = L                  # Initial Length (L_i)
        self.R_outer = R_outer      # Outer Radius (R_i)
        self.t_w = t_w              # Wall Thickness (t_w,i)
        self.t_s = t_s              # Septum Thickness (t_s)
        self.material = material    # Material object
        
        # Derived Geometric Parameters
        self.R_inner = R_outer - t_w
        if self.R_inner <= 0:
            raise ValueError(f"Inner radius must be positive. R_outer={R_outer}, t_w={t_w}")
        
        # Common Parameters from Lab Manual
        self.epsilon_pre = 0.15
        self.L_pre = self.L * (1 + self.epsilon_pre) # Pre-strained length
        
        # Channel Geometry (Common)
        self.r_c = 0.005 # 5.0 mm
        # self.t_s is now passed in
        
        # Actuation Model parameters
        self.A_c = (np.pi * self.r_c**2) / 2
        self.y_c = (self.r_c + self.t_s/2) + (4 * self.r_c) / (3 * np.pi)
        
        # Structural Rigidity (Bending Stiffness)
        # EI = E * I
        # E comes from the material (Young's Modulus)
        # I comes from the geometry (Second Moment of Area)
        self.E_tan = self.material.get_tangent_modulus(self.epsilon_pre)
        self.I = (np.pi / 4) * (self.R_outer**4 - self.R_inner**4)
        self.EI = self.E_tan * self.I # Total resistance to bending

# --- 3. Soft Robot Model Class ---
class SoftRobotModel:
    def __init__(self, segments):
        self.segments = segments
        self.segment_results = []
        
    def calculate_forward_kinematics(self, P1, P2, P3, P4):
        """
        Step 1: INPUT -> Pressures (P1, P2, P3, P4)
        Step 2: PHYSICS -> Calculate Bending Moment (M) and Curvature (kappa)
        Step 3: GEOMETRY -> Calculate shape (Arc)
        Step 4: OUTPUT -> Tip Position (X, Y, Z) and Orientation
        
        Args:
            P1..P4: Input pressures in Pascals (Pa).
        """
        T_total = np.identity(4) # Start at origin (Identity Matrix)
        self.segment_results = []
        
        # Loop through each segment and add its deformation to the chain
        for i, seg in enumerate(self.segments):
            # 1. Calculate Moments (Turning Force) based on Pressure
            # Pressure * Area * Distance = Moment
            # X-Axis Moment comes from P1 (positive) and P3 (negative)
            # Y-Axis Moment comes from P2 (positive) and P4 (negative)
            M_x = (P1 - P3) * seg.A_c * seg.y_c
            M_y = (P2 - P4) * seg.A_c * seg.y_c
            
            # Resultant moment magnitude and direction
            M = np.sqrt(M_x**2 + M_y**2)
            phi = np.arctan2(M_y, M_x)  # Bending direction angle in X-Y plane
            
            # Curvature and arc angle
            kappa = M / seg.EI
            theta = kappa * seg.L_pre
            
            # Transformation matrix for 3D bending
            if np.abs(kappa) < 1e-6:
                # Straight segment
                T_i = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, seg.L_pre],
                    [0, 0, 0, 1]
                ])
            else:
                rho = 1.0 / kappa
                
                # First: Rotation about Z by phi (align bending plane)
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                
                # PCC bend in the aligned plane (bend is in local X-Z)
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                # Combined transformation: Rz(phi) * PCC_bend * Rz(-phi)
                # But simpler: bend in direction phi
                # Using the composite transformation:
                
                # The bending creates an arc. We need to:
                # 1. Rotate to align with bending direction (phi around Z)
                # 2. Apply PCC transformation (bend in X-Z plane)
                # 3. Rotate back? No, we stay in the rotated frame.
                
                # Actually, for simplicity: the bend happens in the plane defined by phi
                # The transformation is:
                # - Arc center at distance rho in direction phi (in X-Y plane)
                # - Arc of angle theta
                
                # Let's use a direct formula:
                # Translation in X-Y: rho*(1-cos(theta)) in direction phi
                # Translation in Z: rho*sin(theta)
                # Rotation: theta about axis perpendicular to phi
                
                # Simplified: bend direction is phi, so rotation axis is (-sin(phi), cos(phi), 0)
                # But PCC assumes bend in X-Z plane. Let me construct it properly.
                
                # Standard PCC in X-Z plane:
                T_pcc = np.array([
                    [cos_theta, 0, sin_theta, rho * (1 - cos_theta)],
                    [0, 1, 0, 0],
                    [-sin_theta, 0, cos_theta, rho * sin_theta],
                    [0, 0, 0, 1]
                ])
                
                # Rotation about Z by phi
                R_z_phi = np.array([
                    [cos_phi, -sin_phi, 0, 0],
                    [sin_phi, cos_phi, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                
                # Apply rotation, then PCC bend, then inverse rotation
                # T_i =R_z(phi) * T_pcc * R_z(-phi)
                R_z_phi_inv = np.array([
                    [cos_phi, sin_phi, 0, 0],
                    [-sin_phi, cos_phi, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                
                T_i = R_z_phi @ T_pcc @ R_z_phi_inv
            
            # Store result with T_base (cumulative transform BEFORE this segment)
            self.segment_results.append({
                'id': i+1,
                'M': M,
                'M_x': M_x,
                'M_y': M_y,
                'phi': phi,
                'kappa': kappa,
                'theta': theta,
                'L_pre': seg.L_pre,
                'rho': 1.0/kappa if np.abs(kappa) > 1e-6 else np.inf,
                'R_outer': seg.R_outer,
                'T_base': T_total.copy() # Store copy of base transform
            })
            
            # Accumulate
            T_total = T_total @ T_i
            
        p_tip = T_total[:3, 3]
        return T_total, p_tip
        
    def validate_model(self):
        """
        Performs physical plausibility checks.
        """
        print("\n--- Section 5: Validation & Error Analysis ---")
        
        # Check 1: Arc Length Conservation
        total_arc_length = sum([res['rho'] * res['theta'] if res['kappa'] > 1e-6 else res['L_pre'] for res in self.segment_results])
        total_L_pre = sum([res['L_pre'] for res in self.segment_results])
        error_arc = abs(total_arc_length - total_L_pre)
        print(f"Check 1: Arc Length Conservation")
        print(f"  Calculated Arc Length: {total_arc_length:.6f} m")
        print(f"  Total Pre-strained Length: {total_L_pre:.6f} m")
        print(f"  Error: {error_arc:.6e} m ({'PASS' if error_arc < 1e-4 else 'FAIL'})")
        
        # Check 2: Maximum Material Strain
        print(f"Check 2: Maximum Material Strain")
        max_strain = 0
        max_strain_seg = -1
        epsilon_pre = 0.15
        
        for res in self.segment_results:
            eps_bend = res['kappa'] * res['R_outer']
            eps_total = epsilon_pre + eps_bend
            if eps_total > max_strain:
                max_strain = eps_total
                max_strain_seg = res['id']
            print(f"  Seg {res['id']}: Total Strain = {eps_total:.4f} (Bend: {eps_bend:.4f})")
            
        print(f"  Max Strain: {max_strain:.4f} in Segment {max_strain_seg}")
        
        # Check 3: Thin-Wall Assumption
        print(f"Check 3: Thin-Wall Assumption (t_w / R < 0.2)")
        for i, seg in enumerate(self.segments):
            ratio = seg.t_w / seg.R_outer
            print(f"  Seg {i+1}: Ratio = {ratio:.3f} ({'PASS' if ratio < 0.2 else 'FAIL'})")

    def compute_workspace(self, num_samples=5000, max_pressure=100000):
        """
        Compute the reachable workspace by sampling random pressure combinations.
        Returns array of reachable tip positions.
        """
        print(f"Computing workspace with {num_samples} samples...")
        workspace_points = []
        
        for i in range(num_samples):
            # Random pressure combinations
            P1 = np.random.uniform(0, max_pressure)
            P2 = np.random.uniform(0, max_pressure)
            P3 = np.random.uniform(0, max_pressure)
            P4 = np.random.uniform(0, max_pressure)
            
            try:
                # Calculate where the tip ends up for these random pressures
                _, p_tip = self.calculate_forward_kinematics(P1, P2, P3, P4)
                
                # Check 1: Did we hit the floor?
                points = self.get_backbone_positions()
                has_ground_collision = min([p[2] for p in points]) < 0
                
                # Check 2: Did segments hit each other?
                has_seg_collision, _ = self.check_segment_collisions()
                
                # Only keep valid (non-colliding) points
                if not has_ground_collision and not has_seg_collision:
                    workspace_points.append(p_tip)
            except:
                pass  # Skip invalid configurations
            
            if (i + 1) % 1000 == 0:
                print(f"  Sampled {i + 1}/{num_samples} configurations...")
        
        print(f"Found {len(workspace_points)} valid configurations")
        return np.array(workspace_points)
    

    def solve_inverse_kinematics_4channel(self, p_target, initial_guess=None, max_iterations=100, tolerance=0.001):
        """
        Solve for all 4 pressures to reach a target 3D position using optimization.
        Uses a Hybrid Approach: Global Search (Differential Evolution) + Local Polishing (L-BFGS-B).
        
        Args:
            p_target: Target [x, y, z] position in meters
            initial_guess: Initial [P1, P2, P3, P4] guess (optional)
            max_iterations: Maximum optimization iterations (for global search)
            tolerance: Convergence tolerance in meters
            
        Returns:
            dict: Solution details including pressures and achieved accuracy.
        """
        try:
            from scipy.optimize import differential_evolution, minimize
        except ImportError:
            # Fallback if the user didn't install the science library
            return {
                'success': False,
                'pressures': [0, 0, 0, 0],
                'tip_position': [0, 0, 0],
                'error': float('inf'),
                'message': 'scipy not installed. Run: pip install scipy'
            }
        
        # --- Inverse Kinematics Logic ---
        # The robot math (FK) goes from Pressure -> Position.
        # We want Position -> Pressure, but the math is too complex to invert directly.
        # So we use an "Optimizer" (AI guesser).
        # We say: "Guess some pressures. Calculate where the tip goes. How far is it from target?"
        # The optimizer keeps guessing better pressures until the error is near zero.
        p_target = np.array(p_target)
        
        # Objective function: minimize distance to target
        # Input 'p_norm' is normalized [0, 1]
        def objective(p_norm):
            # Denormalize
            P1, P2, P3, P4 = p_norm * 100000.0
            try:
                _, p_tip = self.calculate_forward_kinematics(P1, P2, P3, P4)
                error = np.linalg.norm(p_tip - p_target)
                
                # Penalty for using opposing channels (wasteful)
                opposing_penalty = 0.0
                opposing_penalty += min(P1, P3) / 100000.0
                opposing_penalty += min(P2, P4) / 100000.0

                # --- Collision Penalties ---
                collision_penalty = 0.0
                points = self.get_backbone_positions()
                
                # Ground collision (Z < 0)
                # Use a smooth penalty for local optimizer gradient: exp(-Z * large_val) if Z < 0?
                # For now, keep discrete large penalty but only if significantly violating
                min_z = min([p[2] for p in points])
                if min_z < -0.001: # Allow 1mm tolerance for grazing
                    collision_penalty += 10.0 + abs(min_z) * 100
                
                # Segment collision penalty
                has_seg_collision, _ = self.check_segment_collisions()
                if has_seg_collision:
                    collision_penalty += 10.0
                
                return error + 0.05 * opposing_penalty + collision_penalty
            except:
                return 1e6  # Large penalty for invalid configurations
        
        # --- Stage 1: Global Optimization ---
        # Explore the search space to find the valid basin of attraction
        bounds = [(0, 1.0)] * 4
        
        global_result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations,
            atol=tolerance * 1e-3,
            seed=42,
            workers=1,
            updating='deferred',
            disp=False
        )
        
        best_x = global_result.x
        
        # --- Stage 2: Local Polishing ---
        # Use gradient-based method to refine the solution to high precision
        local_result = minimize(
            objective,
            best_x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 50}
        )
        
        # Use the better of the two results
        final_x = local_result.x if local_result.fun < global_result.fun else best_x
        
        # Get final result (Denormalize)
        P1, P2, P3, P4 = final_x * 100000.0
        _, p_tip = self.calculate_forward_kinematics(P1, P2, P3, P4)
        error = np.linalg.norm(p_tip - p_target)
        
        # Check for collisions
        points = self.get_backbone_positions()
        has_ground_collision = min([p[2] for p in points]) < -0.001
        has_seg_collision, seg_msg = self.check_segment_collisions()
        
        success = error < tolerance and not has_ground_collision and not has_seg_collision
        
        if success:
            message = f"Success! Reached target within {error*1000:.2f}mm"
        elif has_ground_collision:
            message = "Convergence failed: solution causes GROUND COLLISION"
        elif has_seg_collision:
            message = f"Convergence failed: solution causes {seg_msg}"
        else:
            message = f"Convergence failed: error {error*1000:.1f}mm > tolerance {tolerance*1000:.1f}mm"
        
        return {
            'success': success,
            'pressures': [P1, P2, P3, P4],
            'tip_position': p_tip.tolist(),
            'error': error,
            'message': message,
            'iterations': (global_result.nit if hasattr(global_result, 'nit') else 0) + (local_result.nit if hasattr(local_result, 'nit') else 0)
        }

    def get_backbone_positions(self):
        """
        Returns a list of 3D points representing the robot backbone.
        Includes Origin (0,0,0) + Tip of Segment 1 + ... + Tip of Segment 5.
        
        FIXED: Now properly reconstructs transformation matrices with phi rotation
        to match the forward kinematics calculation.
        """
        points = [np.array([0.0, 0.0, 0.0])]
        
        if not self.segment_results:
            return points
            
        T_current = np.identity(4)
        
        for res in self.segment_results:
            # Reconstruct T_i from stored results
            theta = res['theta']
            kappa = res['kappa']
            phi = res['phi']
            L_pre = res['L_pre']
            
            # Check if straight or curved
            if np.abs(kappa) < 1e-6:  # Straight segment
                # Simple translation upward (no rotation needed)
                T_i = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, L_pre],
                    [0, 0, 0, 1]
                ])
            else:  # Curved segment
                rho = 1.0 / kappa
                
                # Build rotation matrices
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                # Standard PCC transformation in X-Z plane
                T_pcc = np.array([
                    [cos_theta, 0, sin_theta, rho * (1 - cos_theta)],
                    [0, 1, 0, 0],
                    [-sin_theta, 0, cos_theta, rho * sin_theta],
                    [0, 0, 0, 1]
                ])
                
                # Rotation about Z by phi
                R_z_phi = np.array([
                    [cos_phi, -sin_phi, 0, 0],
                    [sin_phi, cos_phi, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                
                # Rotation about Z by -phi
                R_z_phi_inv = np.array([
                    [cos_phi, sin_phi, 0, 0],
                    [-sin_phi, cos_phi, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                
                # Apply the same transformation as in forward kinematics
                T_i = R_z_phi @ T_pcc @ R_z_phi_inv
            
            T_current = T_current @ T_i
            points.append(T_current[:3, 3])
            
        return points

    def _segment_distance(self, p1, p2, p3, p4):
        """
        Calculate minimum distance between two 3D line segments.
        Segment 1: from p1 to p2
        Segment 2: from p3 to p4
        
        Returns:
            float: Minimum distance between the two segments
        """
        # Direction vectors
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, r)
        e = np.dot(d2, r)
        
        denom = a * c - b * b
        
        # Check if segments are parallel
        if abs(denom) < 1e-10:
            # Parallel segments - use point-to-segment distance
            s = 0.0
            t = d / a if a > 1e-10 else 0.0
        else:
            s = (b * d - a * e) / denom
            t = (c * d - b * e) / denom
        
        # Clamp s and t to [0, 1]
        s = max(0.0, min(1.0, s))
        t = max(0.0, min(1.0, t))
        
        # Calculate closest points
        closest1 = p1 + t * d1
        closest2 = p3 + s * d2
        
        return np.linalg.norm(closest1 - closest2)

    def check_segment_collisions(self, threshold_multiplier=1.5):
        """
        Check for collisions between non-adjacent segments.
        
        Args:
            threshold_multiplier: Safety factor for collision threshold
                                 (1.5 means collision if distance < 1.5 * sum_of_radii)
        
        Returns:
            (bool, str): (has_collision, collision_message)
        """
        if not self.segment_results or len(self.segment_results) < 3:
            return False, ""
        
        # Get backbone positions
        points = self.get_backbone_positions()
        
        # Check all pairs of non-adjacent segments
        for i in range(len(self.segments)):
            for j in range(i + 2, len(self.segments)):  # j > i+1 ensures non-adjacent
                # Get segment endpoints
                p1_start = points[i]
                p1_end = points[i + 1]
                p2_start = points[j]
                p2_end = points[j + 1]
                
                # Calculate minimum distance
                dist = self._segment_distance(p1_start, p1_end, p2_start, p2_end)
                
                # Calculate collision threshold (sum of radii with safety factor)
                r1 = self.segments[i].R_outer
                r2 = self.segments[j].R_outer
                threshold = (r1 + r2) * threshold_multiplier
                
                # Check collision
                if dist < threshold:
                    return True, f"Segments {i+1} and {j+1} too close ({dist*1000:.1f}mm < {threshold*1000:.1f}mm)"
        
        return False, ""

# --- Data: Material Properties ---
MATERIALS_DB = {
    '1': {
        'name': 'EcoFlex 0050',
        'K': 50.0 * 1e6,
        'mu_initial': 0.125 * 1e6, # 125 kPa
        'params': {
            'mu_neo': 0.125 * 1e6, # Using initial mu for Neo-Hookean base
            'C10': 0.050 * 1e6,
            'C01': 0.0125 * 1e6,
            'mu1': 0.125 * 1e6,
            'alpha1': 2.0
        }
    },
    '2': {
        'name': 'Dragon Skin 30',
        'K': 100.0 * 1e6,
        'mu_initial': 0.335 * 1e6, # 335 kPa
        'params': {
            'mu_neo': 0.335 * 1e6,
            'C10': 0.120 * 1e6,
            'C01': 0.0475 * 1e6,
            'mu1': 0.335 * 1e6,
            'alpha1': 2.0
        }
    },
    '3': {
        'name': 'Sylgard 184',
        'K': 200.0 * 1e6,
        'mu_initial': 0.500 * 1e6, # 500 kPa
        'params': {
            'mu_neo': 0.500 * 1e6,
            'C10': 0.200 * 1e6,
            'C01': 0.050 * 1e6,
            'mu1': 0.500 * 1e6,
            'alpha1': 2.0
        }
    }
}

MODELS_DB = {
    '1': 'Neo-Hookean',
    '2': 'Mooney-Rivlin',
    '3': 'Ogden'
}

def get_user_selection():
    print("\n--- Soft Robot Configuration ---")
    print("Select Material:")
    print("1. EcoFlex 0050")
    print("2. Dragon Skin 30")
    print("3. Sylgard 184")
    mat_choice = input("Enter number (1-3): ").strip()
    if mat_choice not in MATERIALS_DB:
        print("Invalid choice, defaulting to EcoFlex 0050")
        mat_choice = '1'
        
    print("\nSelect Constitutive Model:")
    print("1. Neo-Hookean")
    print("2. Mooney-Rivlin")
    print("3. Ogden")
    model_choice = input("Enter number (1-3): ").strip()
    if model_choice not in MODELS_DB:
        print("Invalid choice, defaulting to Neo-Hookean")
        model_choice = '1'
        
    return MATERIALS_DB[mat_choice], MODELS_DB[model_choice]

# --- 7. Session Logging ---
class DualLogger:
    def __init__(self, filename="simulation_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes immediately
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    # Start Session Logging
    sys.stdout = DualLogger()
    print(f"--- Soft Robot Simulation Session Started: {datetime.datetime.now()} ---")
    
    # User Selection
    mat_data, model_name = get_user_selection()
    
    print(f"\nConfiguration: {mat_data['name']} with {model_name} Model")
    
    # Instantiate Material
    material = Material(
        name=mat_data['name'],
        K=mat_data['K'],
        mu_initial=mat_data['mu_initial'],
        params=mat_data['params'],
        model_type=model_name
    )
    
    # Validation of Material Model
    E_approx = 3 * material.mu_initial
    E_full = (9 * material.K * material.mu_initial) / (3 * material.K + material.mu_initial)
    diff_percent = abs(E_approx - E_full) / E_full * 100
    print(f"Material Validation:")
    print(f"  E_approx (3*mu): {E_approx:.2f} Pa")
    print(f"  E_full: {E_full:.2f} Pa")
    print(f"  Difference: {diff_percent:.4f}% (< 2% required)")
    
    # Define Segments (Distinct Geometries)
    # Ask user for number of segments
    while True:
        try:
            num_segs_str = input("\nEnter number of segments (1-100): ").strip()
            num_segs = int(num_segs_str)
            if 1 <= num_segs <= 100:
                break
            print("Please enter a number between 1 and 100.")
        except ValueError:
            print("Invalid number.")

    print(f"Creating robot with {num_segs} segments...")
    segments = []
    # Default geometry for CLI demo
    defaults = {'L': 0.040, 'R_outer': 0.008, 't_w': 0.0012, 't_s': 0.0008}
    
    for i in range(num_segs):
        if i < 5:
            # Keep some variety for the first few to show it works, or just make them uniform?
            # Let's make them uniform for simplicity in CLI, or use the old values for first 5 if we want.
            # But the user asked for "user can define" -> usually implies uniform or custom.
            # In CLI, easier to just make them uniform defaults to avoid asking 100 questions.
            pass
        
        # Create segment with default properties
        seg = Segment(L=defaults['L'], R_outer=defaults['R_outer'], 
                     t_w=defaults['t_w'], t_s=defaults['t_s'], material=material)
        segments.append(seg)
    
    robot = SoftRobotModel(segments)
    
    # Forward Kinematics
    print("\n--- Input Actuation Pressures ---")
    print("Valid Range: 0 to 120000 Pa (approx 1.2 atm)")
    
    def get_valid_pressure(label):
        while True:
            try:
                # IMPORTANT: For DualLogger to capture INPUT prompts, we must print them first.
                # 'input()' writes prompt to stdout/stderr directly, bypassing sys.stdout.
                # So we print the prompt with no newline, then call input() empty.
                print(f"Enter Pressure {label}: ", end='') 
                val_str = input().strip()
                # Log the user's input too so the log is complete!
                print(val_str) 
                
                val = float(val_str)
                if val < 0:
                    print("Pressure cannot be negative. Try again.")
                else:
                    return val
            except ValueError:
                print("Invalid number. Please enter a numeric value.")

    P1 = get_valid_pressure("P1 (0°)")
    P2 = get_valid_pressure("P2 (90°)")
    P3 = get_valid_pressure("P3 (180°)")
    P4 = get_valid_pressure("P4 (270°)")
    
    print(f"\n--- Forward Kinematics (P1={P1}, P2={P2}, P3={P3}, P4={P4}) ---")
    T_total, p_tip = robot.calculate_forward_kinematics(P1, P2, P3, P4)
    
    print("Final Transformation Matrix T_total:")
    print(T_total)
    print(f"Tip Position p_ee: {p_tip} m")
    
    # Validation
    robot.validate_model()
    
    # Inverse Kinematics (4-Channel)
    print("\n--- Section 6: Inverse Kinematics (4-Channel) ---")
    print("\n--- Input Target Position for IK (meters) ---")
    try:
        print("Enter Target X [Default 0.02]: ", end=''); x_in = input().strip(); print(x_in)
        x_t = float(x_in) if x_in else 0.02
        
        print("Enter Target Y [Default 0.01]: ", end=''); y_in = input().strip(); print(y_in)
        y_t = float(y_in) if y_in else 0.01
        
        print("Enter Target Z [Default 0.15]: ", end=''); z_in = input().strip(); print(z_in)
        z_t = float(z_in) if z_in else 0.15
        
        p_target = np.array([x_t, y_t, z_t])
    except ValueError:
        print("Invalid input! Using default target.")
        p_target = np.array([0.02, 0.01, 0.15])
    result = robot.solve_inverse_kinematics_4channel(p_target, tolerance=0.002, max_iterations=200)
    
    print(f"Target: {p_target}")
    print(f"Success: {result['success']}")
    print(f"Pressures: P1={result['pressures'][0]:.0f}, P2={result['pressures'][1]:.0f}, P3={result['pressures'][2]:.0f}, P4={result['pressures'][3]:.0f} Pa")
    print(f"Achieved: {result['tip_position']}")
    print(f"Error: {result['error']*1000:.2f} mm")
    print(f"Message: {result['message']}")

# --- Main Execution ---
if __name__ == "__main__":
    import datetime
    main()
