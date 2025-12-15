# Soft Robot Physics Engine (`SR_2.py`): Technical Documentation

This document provides a detailed technical breakdown of the Python implementation for the soft robot physics engine. It covers the mathematical models, class structure, and algorithmic strategies used for Forward and Inverse Kinematics.

## **1. Core Libraries**

*   **`numpy`**: Used for all vector and matrix operations, particularly 4x4 Homogeneous Transformation Matrices used in kinematics.
*   **`scipy.optimize`**: 
    *   `differential_evolution`: Used for global optimization in Inverse Kinematics to find the basin of attraction.
    *   `minimize` (L-BFGS-B): Used for local gradient-based polishing to achieve high-precision convergence.

---

## **2. Class Structure**

The code is organized into three primary classes following an Object-Oriented design pattern.

### **A. `class Material`**
Encapsulates constitutive material models to calculating the stress-strain relationship.

*   **Key Method:** `get_tangent_modulus(epsilon_pre)`
    *   **Purpose:** Computes the Tangent Modulus ($E_{tan}$) at a specific operating point (pre-strain).
    *   **Implementation:** Supports three hyperelastic models:
        1.  **Neo-Hookean:** $E_{tan} = \mu (2\lambda + \lambda^{-2})$
        2.  **Mooney-Rivlin:** Includes $C_{10}$ and $C_{01}$ parameters for higher-order accuracy.
        3.  **Ogden:** Uses parameters $\mu_1, \alpha_1$ for non-polynomial strain stiffening.

### **B. `class Segment`**
Represents a single modular segment of the robot.

*   **Attributes:**
    *   Geometric: Length ($L$), Outer Radius ($R_{outer}$), Wall Thickness ($t_w$).
    *   Dynamics: Channel radius ($r_c$) and offset distance ($y_c$).
*   **Key Calculations:**
    *   **Area Moment of Inertia ($I$):** Calculated for a hollow cylinder: $I = \frac{\pi}{4}(R_{outer}^4 - R_{inner}^4)$.
    *   **Flexural Rigidity ($EI$):** Combines material stiffness and geometry: $EI = E_{tan} \cdot I$.

### **C. `class SoftRobotModel`**
The core physics engine managing the assembly of segments.

#### **1. Forward Kinematics (`calculate_forward_kinematics`)**
Computes the tip position given input pressures ($P_1, P_2, P_3, P_4$).

*   **Model:** Piecewise Constant Curvature (PCC).
*   **Process per Segment:**
    1.  **Moment Calculation:** Computes bending moments induced by pressure assymetry:
        *   $M_x = (P_1 - P_3) \cdot A_c \cdot y_c$
        *   $M_y = (P_2 - P_4) \cdot A_c \cdot y_c$
    2.  **Curvature ($\kappa$):** $\kappa = \frac{\sqrt{M_x^2 + M_y^2}}{EI}$
    3.  **Transformation Matrix:** Constructs a 4x4 matrix representing the arc:
        *   Rotation $R_z(\phi)$ to align with bending plane.
        *   Standard PCC arc transformation in the plane.
        *   Inverse rotation $R_z(-\phi)$.
    4.  **Chain Multiplication:** $T_{total} = T_1 \cdot T_2 \cdot ... \cdot T_5$

#### **2. Inverse Kinematics (`solve_inverse_kinematics_4channel`)**
Solves for pressures $[P_1, P_2, P_3, P_4]$ given a target position $[x, y, z]$.

*   **Strategy:** **Hybrid Optimization** (Global + Local).
*   **Objective Function:**
    *   Minimizes Euclidean distance: $||P_{tip} - P_{target}||$.
    *   **Penalties:**
        *   **Ground Collision:** Large penalty if any backbone point has $Z < -0.001$.
        *   **Self-Collision:** Large penalty if non-adjacent segments intersect.
        *   **Antagonistic Pressure:** Small penalty for simultaneous activation of opposing channels (e.g., $P_1$ and $P_3$) to encourage energy efficiency.
*   **Algorithm Flow:**
    1.  **Global Search:** `differential_evolution` scans the full 0-100kPa space to avoid local minima.
    2.  **Local Polishing:** `minimize` (L-BFGS-B) takes the best global result and refines it using gradient descent to meet the strict tolerance ($< 1$mm).

#### **3. Collision Detection (`check_segment_collisions`)**
*   **Method:** Geometric intersection test between 3D line segments (capsules).
*   **Algorithm:** Computes the shortest distance between two skew lines. If distance $< (Radius_1 + Radius_2) \times SafetyFactor$, a collision is reported.

---

## **3. Execution Flow**

1.  **User Input:** The script prompts for Pressure or Target Coordinate inputs via `stdin`.
2.  **Instantiation:** Objects are created (`Material` -> `Segment`s -> `SoftRobotModel`).
3.  **Solver Execution:**
    *   **Forward:** Direct O(1) computation.
    *   **Inverse:** Iterative optimization process (O(N) iterations).
4.  **Output:** Prints the resulting 4x4 Transformation Matrix and Tip Position.

## **4. Key Variables Glossary**

*   `epsilon_pre`: Pre-strain ratio (manufacturing stretch).
*   `phi`: Azimuthal angle (direction of bend in XY plane).
*   `theta`: Arc angle (magnitude of bend).
*   `rho`: Radius of curvature ($1/\kappa$).
*   `p_tip`: Final $[x, y, z]$ coordinates of the end-effector.
