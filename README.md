# Soft Robot Design Toolbox (Course Project)

A comprehensive 3D visualization and design tool for soft pneumatic robots, featuring a **novel 4-channel actuation system** and a **physics-based simulation engine** powered by first principles.

## 🌟 Key Novelties

This project introduces advanced capabilities beyond standard 2-channel planar simulations:

### 1. 4-Channel 3D Actuation (NOVELTY)
Unlike traditional 2-channel models restricted to 2D bending, this robot uses **four independent air chambers** arranged radially (0°, 90°, 180°, 270°).
- **Omnidirectional Bending**: Bend the robot in any 3D direction ($\phi$) by driving pressure differences between opposing pairs.
- **Stiffness Control**: Co-activate all channels to increase stiffness without changing position—a "force shield" effect impossible in simpler models.

### 2. First-Principles Physics Engine
The simulation discards empirical "tuning factors" in favor of a rigorous continuum mechanics approach:
- **No Magic Numbers**: Behavior emerges purely from material properties (Neo-Hookean, Mooney-Rivlin, Ogden).
- **Variable Stiffness**: Calculates real-time Tangent Modulus ($E_{tan}$) as the material stretches.
- **Accurate Moments**: $M_{bending} = (P_{high} - P_{low}) \times A_{c} \times y_{c}$

---

## 🚀 Features

### Core Analysis Tools
- **Inverse Kinematics Solver**: Automatically calculates the 4 pressure inputs needed to reach any 3D target coordinate $(x, y, z)$. Uses a hybrid Global (Differential Evolution) + Local (L-BFGS-B) optimization strategy.
- **Parametric Sweep Analysis**: Perform sensitivity analysis on design variables (e.g., "How does Wall Thickness affect max bending?") and generate plots instantly.
- **Workspace Analysis**: Monte Carlo simulation to generate and visualize the complete reachable 3D point cloud of the robot's end-effector.
- **Automated Reporting**: Generate professional engineering reports (`.txt`) detailing every geometric and physical parameter of your current design.

### Visualization & Interaction
- **Real-Time 3D Rendering**: Volumetric styling with accurate tube diameters and seamless segment connectivity.
- **Safety Systems**: 
    - **Self-Collision Detection**: Prevents the robot from intersecting itself.
    - **Ground Plane Protection**: Alerts if the robot is driven into the floor ($Z < 0$).
- **Modern GUI**: Dark-themed, professional interface built with `customtkinter`.

---

## Installation

### Prerequisites
- Python 3.8 or higher

### Install Dependencies
Run the following command to install the required libraries:
```bash
pip install numpy matplotlib customtkinter scipy
```
*Note: `scipy` is required for the Inverse Kinematics solver.*

---

##  How to Use

### Quick Start
1. Run the application:
   ```bash
   python SoftRobotApp.py
   ```
2. The GUI will launch with a default 5-segment robot.

### Using the Novel 4-Channel Actuation
To control the robot's bending in 3D space:
1. Locate the **"4-Channel Actuation"** section in the left sidebar.
2. Understanding the Layout:
   - **P1 (0°)** & **P3 (180°)**: Control bending in the X-Z plane. Increasing P1 bends it one way; P3 the other.
   - **P2 (90°)** & **P4 (270°)**: Control bending in the Y-Z plane.
3. **Try this**: 
   - Set **P1 = 50 kPa** (Robot bends "Forward").
   - Now set **P2 = 50 kPa** (Robot bends "Diagonally").
   - This vector summation proves the 4-channel capability.

### Using the Inverse Kinematics Solver
Let the math do the work for you:
1. Click the **"Inverse Kinematics Solver"** button (bottom left).
2. A new window will open.
3. Enter your target 3D coordinates (e.g., $X=0.05, Y=0.05, Z=0.15$).
4. Click **"Solve"**.
   - The engine searches thousands of pressure combinations.
5. If successful, it displays the 4 required pressures and you can apply these presssures dynamically to the current simulation.
6. (Optional) Manually enter these pressures into the main slider to verify the result.

### 3D Workspace Visualization
Visualise the theoretical "reach" of your robot design:
1. Click **"Workspace Visualization"** (bottom left button).
2. A configuration window appears.
3. Default settings (5000 samples, 100kPa max pressure) are usually fine.
4. Click **"Compute & Visualize"**.
   - The engine simulates 5000 random pressure combinations.
   - It filters out collisions and invalid states.
5. **Result**: A cloud of yellow dots appears in the 3D plot, representing every reachable coordinate for your current robot geometry.

### Design Parametric Sweep
1. Click **"Parametric Sweep"**.
2. Choose a parameter to investigate (e.g., **"Radius (R)"**).
3. Set the range (e.g., 0.005 to 0.015) and number of steps.
4. Click **"Run Sweep"**.
5. A plot will appear showing how that parameter affects the robot's tip height (Z).

---

## 📂 Project Structure

- `SoftRobotApp.py`: **Main Application**. The entry point for the GUI.
- `SR_2.py`: **Physics Engine**. Contains the math for `Material`, `Segment`, and `SoftRobotModel`.
- `README.md`: This file.

## 🎓 Credits
**Course**: EE631[A] — Soft Robotics: Design & Control (2025-26:Semester 1)
**Project**: Development of Comprehensive Simulation Framework for Multi-Channel Soft Continuum Robots
**Date**: December 2025
