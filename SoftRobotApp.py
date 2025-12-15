import customtkinter as ctk # GUI Library
import tkinter as tk # GUI Library and interaction
from tkinter import messagebox, filedialog # GUI interaction
import matplotlib.pyplot as plt # Plotting
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError:
    # Fallback for older Matplotlib versions
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg as NavigationToolbar2Tk
import numpy as np # Math
import datetime # Date and time

# Import Physics Engine
from SR_2 import Material, Segment, SoftRobotModel, MATERIALS_DB, MODELS_DB

# Set Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SoftRobotApp(ctk.CTk):
    """
    Main Application Class.
    Inherits from ctk.CTk (CustomTkinter), which gives us the main window.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Setup the window basics (Title, Size)
        self._setup_window()
        
        # 2. logical variables (Variables that hold our data like Pressure, Material)
        self._init_variables()
        
        # 3. Create the visual layout (Buttons, Sliders, Plots)
        self._setup_layout()
        
        # 4. Draw the empty robot initially
        self.plot_robot([])

    def _setup_window(self):
        self.title("Soft Robot Design Toolbox (Project 2)")
        self.geometry("1200x800")

    def _init_variables(self):
        self.material_var = ctk.StringVar(value="EcoFlex 0050")
        self.model_var = ctk.StringVar(value="Neo-Hookean")
        
        # 4-Channel Pressures
        self.p1_var = ctk.DoubleVar(value=0)
        self.p2_var = ctk.DoubleVar(value=0)
        self.p3_var = ctk.DoubleVar(value=0)
        self.p4_var = ctk.DoubleVar(value=0)
        
        # Segment Data (L, R, t, t_s)
        # Segment Data (L, R, t, t_s)
        self.num_segments_var = ctk.StringVar(value="5")
        
        # Initial defaults
        self.default_seg_params = (0.040, 0.008, 0.0012, 0.0008) # L, R, t_w, t_s for new segments
        
        # Initialize dynamic list of segment variables
        # Structure: List of lists [[L_var, R_var, tw_var, ts_var], ...]
        self.segment_vars = []
        self._initialize_segment_vars(int(self.num_segments_var.get()))

        self.segment_sliders = []
        self.segment_labels = []

        self.workspace_points = None # To store computed workspace points

    def _initialize_segment_vars(self, count):
        """Helper to init variables, preserving old ones if resizing"""
        current_len = len(self.segment_vars)
        
        if count > current_len:
            # Add new segments
            for _ in range(count - current_len):
                # Use default values for new segments
                new_vars = [ctk.DoubleVar(value=v) for v in self.default_seg_params]
                self.segment_vars.append(new_vars)
        elif count < current_len:
            # Trucate list
            self.segment_vars = self.segment_vars[:count]


    def _setup_layout(self):
        """
        Organizes the window using a Grid System (Rows and Columns).
        Column 0: Sidebar (Controls)
        Column 1: Main View (3D Plot)
        """
        self.grid_columnconfigure(1, weight=1) # Main view expands
        self.grid_rowconfigure(0, weight=1)    # Vertical expansion
        self._setup_sidebar()
        self._setup_main_view()

    def _setup_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(self.sidebar_frame, text="Soft Robot\nToolbox", font=ctk.CTkFont(size=24, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))

        # Material Section
        ctk.CTkLabel(self.sidebar_frame, text="Material & Model Configuration", anchor="w").grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        ctk.CTkComboBox(self.sidebar_frame, variable=self.material_var, values=[m['name'] for m in MATERIALS_DB.values()]).grid(row=2, column=0, padx=20, pady=(5, 10), sticky="ew")
        ctk.CTkComboBox(self.sidebar_frame, variable=self.model_var, values=list(MODELS_DB.values())).grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")

        # Geometry Section
        ctk.CTkLabel(self.sidebar_frame, text="Segment Geometry", anchor="w").grid(row=4, column=0, padx=20, pady=(0, 5), sticky="w")
        
        # Segment Count Control
        count_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        count_frame.grid(row=5, column=0, padx=20, pady=5, sticky="ew")
        ctk.CTkLabel(count_frame, text="Count (1-100):").pack(side="left")
        
        self.seg_count_entry = ctk.CTkEntry(count_frame, textvariable=self.num_segments_var, width=50)
        self.seg_count_entry.pack(side="left", padx=5)
        
        def on_count_enter(event=None):
            self._update_segment_count()
            
        self.seg_count_entry.bind('<Return>', on_count_enter)
        ctk.CTkButton(count_frame, text="Set", width=40, command=self._update_segment_count).pack(side="left")

        self.geo_frame = ctk.CTkScrollableFrame(self.sidebar_frame, height=250, fg_color="transparent")
        self.geo_frame.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
        
        self._refresh_segment_controls()

        # Uniform Button
        ctk.CTkButton(self.sidebar_frame, text="Make All Segments Uniform", command=self._make_segments_uniform, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE")).grid(row=7, column=0, padx=20, pady=(10, 5), sticky="ew")

        # Actuation Section
        ctk.CTkLabel(self.sidebar_frame, text="4-Channel Actuation (Pa)", anchor="w").grid(row=8, column=0, padx=20, pady=(20, 5), sticky="w")
        act_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        act_frame.grid(row=9, column=0, padx=20, pady=5, sticky="ew")
        
        self._create_pressure_slider(act_frame, "P1 (0°)", self.p1_var)
        self._create_pressure_slider(act_frame, "P2 (90°)", self.p2_var)
        self._create_pressure_slider(act_frame, "P3 (180°)", self.p3_var)
        self._create_pressure_slider(act_frame, "P4 (270°)", self.p4_var)

        # Action Buttons
        ctk.CTkButton(self.sidebar_frame, text="Generate Report", command=self.generate_report, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE")).grid(row=10, column=0, padx=20, pady=(20, 10), sticky="ew")
        ctk.CTkButton(self.sidebar_frame, text="Parametric Sweep", command=self.open_sweep_window, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE")).grid(row=11, column=0, padx=20, pady=(0, 10), sticky="ew")
        ctk.CTkButton(self.sidebar_frame, text="Inverse Kinematics Solver", command=self.open_ik_window, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE")).grid(row=12, column=0, padx=20, pady=(0, 10), sticky="ew")
        ctk.CTkButton(self.sidebar_frame, text="Workspace Visualization", command=self.visualize_workspace, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE")).grid(row=13, column=0, padx=20, pady=(0, 20), sticky="ew")

    def _update_segment_count(self):
        try:
            count = int(self.num_segments_var.get())
            if count < 1: count = 1
            if count > 100: count = 100
            self.num_segments_var.set(str(count))
            
            self._initialize_segment_vars(count)
            self._refresh_segment_controls()
            self.run_analysis()
        except ValueError:
            pass

    def _refresh_segment_controls(self):
        # Clear existing
        for widget in self.geo_frame.winfo_children():
            widget.destroy()
        
        self.segment_sliders = []
        self.segment_labels = []
        
        # Populate new
        for i in range(len(self.segment_vars)):
            self._create_segment_controls(i)

    def _create_segment_controls(self, i):
        seg_frame = ctk.CTkFrame(self.geo_frame, fg_color=("gray80", "gray20"))
        seg_frame.pack(fill="x", pady=5, padx=5)
        
        # Minimizable header for segments? For 100 segments, this is going to be HUGE.
        # But for now, standard implementation.
        ctk.CTkLabel(seg_frame, text=f"Segment {i+1}", font=("Arial", 12, "bold")).pack(anchor="w", padx=5, pady=2)
        
        self.segment_sliders.append([])
        self.segment_labels.append([])
        
        self._add_slider_row(seg_frame, "L", self.segment_vars[i][0], 0.020, 0.060, 0.001, i)
        self._add_slider_row(seg_frame, "R", self.segment_vars[i][1], 0.005, 0.015, 0.0005, i)
        self._add_slider_row(seg_frame, "t_w", self.segment_vars[i][2], 0.0005, 0.0020, 0.0001, i)
        self._add_slider_row(seg_frame, "t_s", self.segment_vars[i][3], 0.0001, 0.0020, 0.0001, i)

    def _add_slider_row(self, parent, label, variable, from_, to_, step, seg_idx):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=2)
        ctk.CTkLabel(frame, text=label, width=30, anchor="w").pack(side="left")
        
        val_lbl = ctk.CTkLabel(frame, text=f"{variable.get():.4f}", width=50, anchor="e")
        val_lbl.pack(side="right", padx=5)
        
        def update_val(v):
            variable.set(v)
            val_lbl.configure(text=f"{v:.4f}")
            self.run_analysis()
            
        slider = ctk.CTkSlider(frame, from_=from_, to=to_, number_of_steps=100, command=update_val)
        slider.set(variable.get())
        slider.pack(side="right", fill="x", expand=True, padx=5)
        
        self.segment_sliders[seg_idx].append(slider)
        self.segment_labels[seg_idx].append(val_lbl)

    def _create_pressure_slider(self, parent, label, variable):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=2)
        ctk.CTkLabel(frame, text=label, width=50, anchor="w").pack(side="left")
        val_lbl = ctk.CTkLabel(frame, text=f"{int(variable.get())}", width=50, anchor="e")
        val_lbl.pack(side="right", padx=5)
        
        # Trace variable to update label automatically
        def update_label(*args):
            val_lbl.configure(text=f"{int(variable.get())}")
        variable.trace_add("write", update_label)
        
        def slider_event(v):
            self.run_analysis()
            
        # Bind variable to slider so it updates when variable changes
        slider = ctk.CTkSlider(frame, from_=0, to=100000, number_of_steps=100, variable=variable, command=slider_event)
        slider.pack(side="right", fill="x", expand=True, padx=5)

    def _setup_main_view(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        plt.style.use('dark_background')
        self.fig = plt.Figure(figsize=(5, 5), dpi=100, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#2b2b2b')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add Navigation Toolbar
        self.toolbar_frame = ctk.CTkFrame(self.main_frame, height=40)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        # The toolbar packs itself into 'window' argument (toolbar_frame) automatically
        
        self.status_frame = ctk.CTkFrame(self.main_frame, height=100)
        self.status_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.result_label = ctk.CTkLabel(self.status_frame, text="Status: Ready", font=ctk.CTkFont(size=16))
        self.result_label.pack(pady=20, padx=20)

    def _make_segments_uniform(self):
        # Set all segments to the specific default values requested
        # L=0.030, R=0.007, t=0.001, t_s=0.0008
        target_vals = [0.030, 0.007, 0.0010, 0.0008]
        
        for i in range(len(self.segment_vars)):
            for j in range(4):
                self.segment_vars[i][j].set(target_vals[j])
                # Provide feedback if widget exists
                if i < len(self.segment_labels) and j < len(self.segment_labels[i]):
                     self.segment_labels[i][j].configure(text=f"{target_vals[j]:.4f}")
                     self.segment_sliders[i][j].set(target_vals[j])
        
        self.run_analysis()

    def get_material_obj(self):
        name = self.material_var.get()
        mat_data = next((v for v in MATERIALS_DB.values() if v['name'] == name), None)
        if not mat_data: return None
        return Material(mat_data['name'], mat_data['K'], mat_data['mu_initial'], mat_data['params'], self.model_var.get())

    def build_robot(self):
        """
        Helper function: Reads all sliders and creates the 'SoftRobotModel' object.
        This is how we pass data from GUI -> Physics Engine.
        """
        material = self.get_material_obj()
        segments = [Segment(v[0].get(), v[1].get(), v[2].get(), v[3].get(), material) for v in self.segment_vars]
        return SoftRobotModel(segments)

    def run_analysis(self):
        """
        CORE FUNCTION: Triggered whenever a slider moves.
        1. Build the robot model from GUI inputs.
        2. Run Physics (Forward Kinematics).
        3. Check for Collisions.
        4. Update the 3D Plot.
        """
        try:
            robot = self.build_robot()
            
            # --- PHYSICS CALCULATION ---
            T_total, p_tip = robot.calculate_forward_kinematics(
                self.p1_var.get(), self.p2_var.get(), self.p3_var.get(), self.p4_var.get()
            )
            points = robot.get_backbone_positions()
            
            # Pass the current robot to check for collisions
            collision_msg = self._check_collision(robot, points, p_tip)
            
            if collision_msg:
                # Show error in red if invalid
                self.result_label.configure(text=f"BLOCKED: {collision_msg}", text_color="red")
            else:
                # Save valid result and update plot
                self.last_robot = robot
                self.last_tip = p_tip
                self.plot_robot(points)
                self.result_label.configure(text=f"Tip Position: X={p_tip[0]:.4f} m, Y={p_tip[1]:.4f} m, Z={p_tip[2]:.4f} m", text_color=("black", "white"))
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _check_collision(self, robot, points, p_tip):
        # Ground collision check
        if min([p[2] for p in points]) < -0.005:
            return "GROUND COLLISION"
        
        # Segment-to-segment collision check using the CURRENT robot
        has_collision, msg = robot.check_segment_collisions()
        if has_collision:
            return f"SEGMENT COLLISION: {msg}"
        
        # Simplified self-collision (tip near origin)
        if np.linalg.norm(p_tip) < 0.02 and p_tip[2] < 0.05 and len(points) > 5:
            return "SELF COLLISION (tip near base)"
        
        return None

    def plot_robot(self, points):
        """
        Visualizes the robot in 3D using Matplotlib.
        It clears the old plot and draws the new segments as 3D tubes.
        """
        self.ax.clear()
        
        # Calculate dynamic limits
        max_dim = 0.25 # Default minimum
        if hasattr(self, 'last_robot') and self.last_robot.segment_results:
             # Crude estimate: sum of lengths
             total_len = sum([s.L for s in self.last_robot.segments])
             max_dim = max(max_dim, total_len * 0.8) # Heuristic
        
        # Better estimate using points if available
        if len(points) > 0:
             # Points format: list of arrays
             all_p = np.array(points)
             # Max range in X, Y, Z
             max_range = np.max(np.abs(all_p))
             max_dim = max(max_dim, max_range * 1.2) # Add 20% buffer

        self._setup_plot_axes(limit=max_dim)
        
        # Draw the Ground Plane (Grid)
        xx, yy = np.meshgrid(np.linspace(-max_dim, max_dim, 10), np.linspace(-max_dim, max_dim, 10))
        self.ax.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.2)

        # Draw Segments (if they exist)
        if hasattr(self, 'last_robot') and self.last_robot.segment_results:
            colors = ['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FF8800']
            for idx, res in enumerate(self.last_robot.segment_results):
                # _get_segment_mesh creates the 3D tube shape for this segment
                X, Y, Z = self._get_segment_mesh(res)
                # Cycle through colors
                color = colors[idx % len(colors)]
                self.ax.plot_surface(X, Y, Z, color=color, alpha=0.75, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True)

        # Plot workspace points if they exist (Yellow dots)
        if self.workspace_points is not None and len(self.workspace_points) > 0:
            self.ax.scatter(self.workspace_points[:, 0], self.workspace_points[:, 1], self.workspace_points[:, 2], c='yellow', s=1, alpha=0.5, label='Reachable Workspace')
            self.ax.legend(loc='upper left')

        self.canvas.draw() # Refresh the canvas on screen

    def _setup_plot_axes(self, limit=0.25):
        self.ax.set_title("Soft Robot 3D Pose", color='white')
        self.ax.set_xlabel("X (m)", color='white')
        self.ax.set_ylabel("Y (m)", color='white')
        self.ax.set_zlabel("Z (m)", color='white')
        self.ax.tick_params(colors='white')
        
        # Determine strict equal aspect ratio box
        self.ax.set_xlim(-limit, limit) 
        self.ax.set_ylim(-limit, limit) 
        self.ax.set_zlim(0, limit * 2) # Z is usually taller? Or just keep cubic?
        # Actually for soft robots, usually height (Z) is the main dimension.
        # Let's keep it cubic-ish but usually 0 to Z_max.
        # To keep aspect ratio correct with [1,1,1], ranges must be equal size.
        # Range X: 2*limit. Range Y: 2*limit. Range Z needs to be 2*limit size too?
        # Yes, for 1,1,1 aspect. So Z can be 0 to 2*limit? No, that's range 2*limit.
        self.ax.set_zlim(0, 2*limit)
        
        self.ax.set_box_aspect([1,1,1])

    def _get_segment_mesh(self, res):
        R, L, kappa, theta = res['R_outer'], res['L_pre'], res['kappa'], res['theta']
        phi, T_base = res.get('phi', 0), res['T_base']
        
        n_theta, n_long = 24, 20
        theta_vals = np.linspace(0, 2*np.pi, n_theta)
        
        if abs(kappa) < 1e-6: # Straight
            s_vals = np.linspace(0, L, n_long)
            s_grid, theta_grid = np.meshgrid(s_vals, theta_vals, indexing='ij')
            x_local, y_local, z_local = R * np.cos(theta_grid), R * np.sin(theta_grid), s_grid
        else: # Curved
            rho = 1.0 / kappa
            u_vals = np.linspace(0, theta, n_long)
            u_grid, theta_grid = np.meshgrid(u_vals, theta_vals, indexing='ij')
            
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
            x_local = np.zeros_like(u_grid); y_local = np.zeros_like(u_grid); z_local = np.zeros_like(u_grid)
            
            for i in range(n_long):
                u = u_vals[i]
                # Centerline
                xc = rho * (1 - np.cos(u))
                zc = rho * np.sin(u)
                # Rotate centerline by phi
                cx, cy = xc * cos_phi, xc * sin_phi
                
                # Normal vector (Corrected sign)
                nx, ny, nz = -np.cos(u)*cos_phi, -np.cos(u)*sin_phi, np.sin(u)
                bx, by, bz = -sin_phi, cos_phi, 0
                
                for j in range(n_theta):
                    rn, rb = R * np.cos(theta_vals[j]), R * np.sin(theta_vals[j])
                    x_local[i,j] = cx + rn*nx + rb*bx
                    y_local[i,j] = cy + rn*ny + rb*by
                    z_local[i,j] = zc + rn*nz + rb*bz

        # Transform to global
        shape = x_local.shape
        x_g, y_g, z_g = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                pt = T_base @ np.array([x_local[i,j], y_local[i,j], z_local[i,j], 1.0])
                x_g[i,j], y_g[i,j], z_g[i,j] = pt[0], pt[1], pt[2]
                
        return x_g, y_g, z_g

    def generate_report(self):
        if not hasattr(self, 'last_robot'):
            messagebox.showwarning("Warning", "Run analysis first!")
            return
        
        report = f"SOFT ROBOT SIMULATION REPORT\nDate: {datetime.datetime.now()}\n{'='*60}\n\n"
        report += f"CONFIGURATION:\n"
        report += f"  Material: {self.material_var.get()}\n"
        report += f"  Model:    {self.model_var.get()}\n"
        report += f"  Pressures: P1={self.p1_var.get():.0f}, P2={self.p2_var.get():.0f}, P3={self.p3_var.get():.0f}, P4={self.p4_var.get():.0f} Pa\n\n"
        
        report += "SEGMENT GEOMETRY INPUTS:\n"
        report += f"{'ID':<4} {'Length (m)':<12} {'Radius (m)':<12} {'Wall (m)':<12} {'Septum (m)':<12}\n"
        report += "-"*60 + "\n"
        for i, s in enumerate(self.last_robot.segments):
            report += f"{i+1:<4} {s.L:<12.4f} {s.R_outer:<12.4f} {s.t_w:<12.4f} {s.t_s:<12.4f}\n"

        report += "\n\nPHYSICS ENGINE RESULTS (PER SEGMENT):\n"
        report += f"{'ID':<4} {'Moment (Nm)':<12} {'Curve (1/m)':<12} {'Angle (rad)':<12} {'Direction (deg)':<16} {'Radius (m)':<12}\n"
        report += "-"*80 + "\n"
        
        if self.last_robot.segment_results:
            for res in self.last_robot.segment_results:
                deg_phi = np.degrees(res['phi'])
                report += f"{res['id']:<4} {res['M']:<12.6f} {res['kappa']:<12.4f} {res['theta']:<12.4f} {deg_phi:<16.2f} {res['rho']:<12.4f}\n"
        
        report += f"\n\nFINAL TIP POSITION:\n"
        report += f"  X: {self.last_tip[0]:.6f} m\n"
        report += f"  Y: {self.last_tip[1]:.6f} m\n"
        report += f"  Z: {self.last_tip[2]:.6f} m\n"

        # --- VALIDATION CHECKS (Matches CLI Output) ---
        report += "\n" + "="*60 + "\n"
        report += "VALIDATION & ERROR ANALYSIS:\n\n"
        
        if self.last_robot.segment_results:
            # Check 1: Arc Length Conservation
            total_arc_length = sum([res['rho'] * res['theta'] if res['kappa'] > 1e-6 else res['L_pre'] for res in self.last_robot.segment_results])
            total_L_pre = sum([res['L_pre'] for res in self.last_robot.segment_results])
            error_arc = abs(total_arc_length - total_L_pre)
            
            report += "Check 1: Arc Length Conservation\n"
            report += f"  Calculated Arc Length: {total_arc_length:.6f} m\n"
            report += f"  Total Pre-strained Length: {total_L_pre:.6f} m\n"
            report += f"  Error: {error_arc:.6e} m ({'PASS' if error_arc < 1e-4 else 'FAIL'})\n\n"

            # Check 2: Maximum Material Strain
            report += "Check 2: Maximum Material Strain\n"
            max_strain = 0
            max_strain_seg = -1
            epsilon_pre = 0.15
            
            for res in self.last_robot.segment_results:
                eps_bend = res['kappa'] * res['R_outer']
                eps_total = epsilon_pre + eps_bend
                if eps_total > max_strain:
                    max_strain = eps_total
                    max_strain_seg = res['id']
                report += f"  Seg {res['id']}: Total Strain = {eps_total:.4f} (Bend: {eps_bend:.4f})\n"
            report += f"  Max Strain: {max_strain:.4f} in Segment {max_strain_seg}\n\n"
            
            # Check 3: Thin-Wall Assumption
            report += "Check 3: Thin-Wall Assumption (t_w / R < 0.2)\n"
            for i, seg in enumerate(self.last_robot.segments):
                ratio = seg.t_w / seg.R_outer
                report += f"  Seg {i+1}: Ratio = {ratio:.3f} ({'PASS' if ratio < 0.2 else 'FAIL'})\n"
        
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if filename:
            with open(filename, 'w') as f: f.write(report)
            messagebox.showinfo("Success", "Detailed report saved!")

    def open_sweep_window(self):
        win = ctk.CTkToplevel(self)
        win.title("Parametric Sweep"); win.geometry("450x450")
        
        ctk.CTkLabel(win, text="Select Parameter:", font=("Arial", 14, "bold")).pack(pady=10)
        param_var = ctk.StringVar(value="Length (L)") # Default value
        param_combo = ctk.CTkComboBox(win, variable=param_var, 
                                       values=["Length (L)", "Radius (R)", "Thickness (t_w)", "Septum (t_s)",
                                              "Pressure P1 (0°)", "Pressure P2 (90°)", 
                                              "Pressure P3 (180°)", "Pressure P4 (270°)"])
        param_combo.pack(pady=5)
        
        # Segment selector (only for geometric parameters)
        seg_label = ctk.CTkLabel(win, text="Segment Index:")
        seg_label.pack(pady=5)
        idx_var = ctk.StringVar(value="1")
        
        # Prepare list of segments
        segment_options = ["All Segments"] + [str(i+1) for i in range(len(self.segment_vars))]
        # If too many segments, maybe only show "All Segments" or a subset? 
        # For now, let's limit the combobox dump if > 20
        if len(self.segment_vars) > 20: 
             # If too many, rely on "All Segments" or just first 20. 
             # Or switch to Entry again? 
             # Let's keep it simple for now, user can drag scroll if ctk supports it, or we just clip it.
             # Actually CTkComboBox has a scrollbar.
             pass
             
        seg_combo = ctk.CTkComboBox(win, variable=idx_var, values=segment_options)
        seg_combo.pack(pady=5)
        
        # Function to show/hide segment selector based on parameter type
        def update_visibility(*args):
            if param_var.get().startswith("Pressure"):
                seg_label.pack_forget()
                seg_combo.pack_forget()
            else:
                seg_label.pack(pady=5)
                seg_combo.pack(pady=5)
        
        param_var.trace('w', update_visibility)
        
        ctk.CTkLabel(win, text="Range (Start, End, Steps):").pack(pady=5)
        rf = ctk.CTkFrame(win, fg_color="transparent"); rf.pack()
        s_var, e_var, st_var = ctk.DoubleVar(value=0.01), ctk.DoubleVar(value=0.10), ctk.IntVar(value=20)
        
        # Update default range based on parameter type
        def update_range(*args):
            if param_var.get().startswith("Pressure"):
                s_var.set(0)
                e_var.set(100000)
                st_var.set(20)
            elif param_var.get() == "Length (L)":
                s_var.set(0.02)
                e_var.set(0.06)
                st_var.set(20)
            elif param_var.get() == "Radius (R)":
                s_var.set(0.005)
                e_var.set(0.015)
                st_var.set(20)
            elif param_var.get() == "Thickness (t_w)":
                s_var.set(0.0005)
                e_var.set(0.002)
                st_var.set(20)
            elif param_var.get() == "Septum (t_s)":
                s_var.set(0.0001)
                e_var.set(0.002)
                st_var.set(20)
        
        param_var.trace('w', update_range)
        
        for v in [s_var, e_var, st_var]: ctk.CTkEntry(rf, textvariable=v, width=60).pack(side="left", padx=5)
        
        def run():
            param_name = param_var.get()
            vals = np.linspace(s_var.get(), e_var.get(), st_var.get())
            z_defs = []
            base = self.build_robot()
            
            # Determine if sweeping geometry or pressure
            is_pressure = param_name.startswith("Pressure")
            
            if is_pressure:
                # Pressure sweep
                pressure_map = {
                    "Pressure P1 (0°)": 0,
                    "Pressure P2 (90°)": 1,
                    "Pressure P3 (180°)": 2,
                    "Pressure P4 (270°)": 3
                }
                p_idx = pressure_map[param_name]
                p_name = f"P{p_idx+1}"
                
                for v in vals:
                    # Get current pressures
                    pressures = [self.p1_var.get(), self.p2_var.get(), 
                                self.p3_var.get(), self.p4_var.get()]
                    # Modify the target pressure
                    pressures[p_idx] = v
                    
                    try:
                        _, tip = base.calculate_forward_kinematics(*pressures)
                        # Check for ground collision
                        points = base.get_backbone_positions()
                        if min([p[2] for p in points]) < 0:
                            z_defs.append(np.nan)  # Invalid configuration
                        else:
                            z_defs.append(tip[2])
                    except:
                        z_defs.append(np.nan)
            else:
                # Geometry sweep
                seg_selection = idx_var.get()
                all_segments = (seg_selection == "All Segments")
                seg_idx = -1 if all_segments else int(seg_selection) - 1
                
                param_map = {"Length (L)": "L", "Radius (R)": "R_outer", "Thickness (t_w)": "t_w", "Septum (t_s)": "t_s"}
                p_name = param_map[param_name]
                
                for v in vals:
                    # Create segments with the modified parameter value
                    segs = []
                    for i, s in enumerate(base.segments):
                        if all_segments or i == seg_idx:
                            # Create segment with modified parameter
                            if p_name == "L":
                                segs.append(Segment(v, s.R_outer, s.t_w, s.t_s, s.material))
                            elif p_name == "R_outer":
                                segs.append(Segment(s.L, v, s.t_w, s.t_s, s.material))
                            elif p_name == "t_w":
                                segs.append(Segment(s.L, s.R_outer, v, s.t_s, s.material))
                            elif p_name == "t_s":
                                segs.append(Segment(s.L, s.R_outer, s.t_w, v, s.material))
                        else:
                            # Keep original segment parameters
                            segs.append(Segment(s.L, s.R_outer, s.t_w, s.t_s, s.material))
                    
                    try:
                        robot = SoftRobotModel(segs)
                        _, tip = robot.calculate_forward_kinematics(self.p1_var.get(), self.p2_var.get(), self.p3_var.get(), self.p4_var.get())
                        # Check for ground collision
                        points = robot.get_backbone_positions()
                        if min([p[2] for p in points]) < 0:
                            z_defs.append(np.nan)  # Invalid configuration
                        else:
                            z_defs.append(tip[2])
                    except: 
                        z_defs.append(np.nan)
                
            # Create enhanced plot
            plt.figure(facecolor='#2b2b2b', figsize=(10, 6))
            ax = plt.axes()
            ax.set_facecolor('#2b2b2b')
            
            # Plot data
            ax.plot(vals, z_defs, 'c-o', linewidth=2, markersize=6)
            
            # Labels and title
            if is_pressure:
                ax.set_xlabel(f'{p_name} (Pa)', color='white', fontsize=12)
                title = f'Parametric Sweep: {p_name}\n'
                title += f'Geometry: All segments with current configuration'
            else:
                param_labels = {"L": "Length (m)", "R_outer": "Radius (m)", "t_w": "Thickness (m)", "t_s": "Septum Thickness (m)"}
                ax.set_xlabel(param_labels.get(p_name, p_name), color='white', fontsize=12)
                
                if all_segments:
                    title = f'Parametric Sweep: {p_name} - All Segments\n'
                else:
                    title = f'Parametric Sweep: {p_name} - Segment {seg_idx+1}\n'
                title += f'Pressures: P1={self.p1_var.get():.0f}, P2={self.p2_var.get():.0f}, P3={self.p3_var.get():.0f}, P4={self.p4_var.get():.0f} Pa'
            
            ax.set_ylabel('Tip Height Z (m)', color='white', fontsize=12)
            ax.set_title(title, color='white', fontsize=14)
            
            ax.tick_params(colors='white')
            ax.grid(True, color='gray', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        ctk.CTkButton(win, text="Run Sweep", command=run).pack(pady=30)

    def open_ik_window(self):
        """Open Inverse Kinematics Solver window"""
        win = ctk.CTkToplevel(self)
        win.title("Inverse Kinematics Solver")
        win.geometry("500x550")
        
        ctk.CTkLabel(win, text="4-Channel Inverse Kinematics Solver", font=("Arial", 16, "bold")).pack(pady=15)
        ctk.CTkLabel(win, text="Enter target position and solve for pressures", font=("Arial", 12)).pack(pady=5)
        
        # Target position inputs
        input_frame = ctk.CTkFrame(win)
        input_frame.pack(pady=15, padx=20, fill="x")
        
        ctk.CTkLabel(input_frame, text="Target Position (meters):", font=("Arial", 13, "bold")).pack(anchor="w", pady=5)
        
        # X input
        x_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        x_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(x_frame, text="X:", width=30).pack(side="left")
        x_var = ctk.StringVar(value="0.05")
        x_entry = ctk.CTkEntry(x_frame, textvariable=x_var, width=100)
        x_entry.pack(side="left", padx=5)

        
        # Y input
        y_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        y_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(y_frame, text="Y:", width=30).pack(side="left")
        y_var = ctk.StringVar(value="0.05")
        y_entry = ctk.CTkEntry(y_frame, textvariable=y_var, width=100)
        y_entry.pack(side="left", padx=5)

        
        # Z input
        z_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        z_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(z_frame, text="Z:", width=30).pack(side="left")
        z_var = ctk.StringVar(value="0.15")
        z_entry = ctk.CTkEntry(z_frame, textvariable=z_var, width=100)
        z_entry.pack(side="left", padx=5)

        
        # Results display
        result_frame = ctk.CTkFrame(win)
        result_frame.pack(pady=15, padx=20, fill="both", expand=True)
        
        result_text = ctk.CTkTextbox(result_frame, height=200, font=("Courier", 11))
        result_text.pack(fill="both", expand=True, padx=5, pady=5)
        result_text.insert("1.0", "Click 'Solve' to find pressures for target position...")
        result_text.configure(state="disabled")
        
        # Store solved pressures
        solved_pressures = [0, 0, 0, 0]
        
        def solve_ik():
            try:
                # Validate and convert inputs
                try:
                    x = float(x_var.get())
                    y = float(y_var.get())
                    z = float(z_var.get())
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid numbers for X, Y, and Z coordinates.")
                    return
                
                target = np.array([x, y, z])
                
                result_text.configure(state="normal")
                result_text.delete("1.0", "end")
                result_text.insert("1.0", f"Solving for target: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] m\n")
                result_text.insert("end", "Running optimization...\n\n")
                result_text.configure(state="disabled")
                win.update()
                
                # Build robot and solve
                robot = self.build_robot()
                result = robot.solve_inverse_kinematics_4channel(target, tolerance=0.002, max_iterations=500)
                
                # Update display
                result_text.configure(state="normal")
                result_text.delete("1.0", "end")
                
                result_text.insert("1.0", f"═══ INVERSE KINEMATICS RESULT ═══\n\n")
                result_text.insert("end", f"Target: [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}] m\n\n")
                
                if result['success']:
                    result_text.insert("end", "✓ SUCCESS!\n\n", "success")
                else:
                    result_text.insert("end", "⚠ WARNING\n\n", "warning")
                
                result_text.insert("end", f"Computed Pressures:\n")
                result_text.insert("end", f"  P1 (0°):   {result['pressures'][0]:>8.0f} Pa\n")
                result_text.insert("end", f"  P2 (90°):  {result['pressures'][1]:>8.0f} Pa\n")
                result_text.insert("end", f"  P3 (180°): {result['pressures'][2]:>8.0f} Pa\n")
                result_text.insert("end", f"  P4 (270°): {result['pressures'][3]:>8.0f} Pa\n\n")
                
                tip = result['tip_position']
                result_text.insert("end", f"Achieved Position:\n")
                result_text.insert("end", f"  [{tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f}] m\n\n")
                
                result_text.insert("end", f"Error: {result['error']*1000:.2f} mm\n")
                result_text.insert("end", f"Iterations: {result.get('iterations', 'N/A')}\n\n")
                result_text.insert("end", f"Status: {result['message']}\n")
                
                result_text.configure(state="disabled")
                
                # Store pressures for apply button
                solved_pressures[0] = result['pressures'][0]
                solved_pressures[1] = result['pressures'][1]
                solved_pressures[2] = result['pressures'][2]
                solved_pressures[3] = result['pressures'][3]
                
            except Exception as e:
                result_text.configure(state="normal")
                result_text.delete("1.0", "end")
                result_text.insert("1.0", f"ERROR: {str(e)}\n\n")
                result_text.insert("end", "Make sure scipy is installed:\n")
                result_text.insert("end", "pip install scipy")
                result_text.configure(state="disabled")
        
        def apply_to_robot():
            if all(p == 0 for p in solved_pressures):
                messagebox.showwarning("No Solution", "Please solve for pressures first!")
                return
            
            # Set the pressure sliders
            self.p1_var.set(solved_pressures[0])
            self.p2_var.set(solved_pressures[1])
            self.p3_var.set(solved_pressures[2])
            self.p4_var.set(solved_pressures[3])
            
            # Trigger analysis
            self.run_analysis()
            
            messagebox.showinfo("Applied", "Pressures applied to robot!")
            win.destroy()
        
        # Buttons
        button_frame = ctk.CTkFrame(win, fg_color="transparent")
        button_frame.pack(pady=10)
        
        ctk.CTkButton(button_frame, text="Solve for Pressures", command=solve_ik, width=150, height=35).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Apply to Robot", command=apply_to_robot, width=150, height=35, fg_color="green").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Close", command=win.destroy, width=100, height=35, fg_color="gray").pack(side="left", padx=5)

    def visualize_workspace(self):
        """Open a window to compute and visualize the robot's workspace."""
        win = ctk.CTkToplevel(self)
        win.title("Workspace Visualization")
        win.geometry("400x350")
        win.transient(self) # Keep on top of main window

        ctk.CTkLabel(win, text="Workspace Computation", font=("Arial", 16, "bold")).pack(pady=15)
        ctk.CTkLabel(win, text="This tool samples random pressures to find the\nreachable points for the current robot design.", justify="center").pack(pady=5)

        # --- Configuration Frame ---
        config_frame = ctk.CTkFrame(win)
        config_frame.pack(pady=15, padx=20, fill="x")

        # Number of samples
        samples_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        samples_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(samples_frame, text="Number of Samples:", width=150, anchor="w").pack(side="left")
        samples_var = ctk.StringVar(value="5000")
        ctk.CTkEntry(samples_frame, textvariable=samples_var, width=100).pack(side="left")

        # Max pressure
        pressure_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        pressure_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(pressure_frame, text="Max Pressure (Pa):", width=150, anchor="w").pack(side="left")
        pressure_var = ctk.StringVar(value="100000")
        ctk.CTkEntry(pressure_frame, textvariable=pressure_var, width=100).pack(side="left")

        # --- Status and Progress ---
        status_label = ctk.CTkLabel(win, text="Status: Ready to compute", text_color="gray")
        status_label.pack(pady=10)

        def run_computation():
            try:
                num_samples = int(samples_var.get())
                max_pressure = float(pressure_var.get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for samples and pressure.", parent=win)
                return

            status_label.configure(text="Computing... this may take a moment.", text_color="cyan")
            win.update_idletasks() # Force GUI update

            robot = self.build_robot()
            # This will print progress to the console
            self.workspace_points = robot.compute_workspace(num_samples=num_samples, max_pressure=max_pressure)

            status_label.configure(text=f"Computation complete. Found {len(self.workspace_points)} points.", text_color="lightgreen")
            messagebox.showinfo("Success", f"Workspace computation finished. Found {len(self.workspace_points)} reachable points.", parent=win)

            # Redraw the main plot to show the workspace
            self.plot_robot(self.last_robot.get_backbone_positions() if hasattr(self, 'last_robot') else [])
            win.destroy()

        ctk.CTkButton(win, text="Compute & Visualize", command=run_computation, height=40).pack(pady=20, padx=20, fill="x")
        ctk.CTkButton(win, text="Close", command=win.destroy, fg_color="gray").pack(pady=5, padx=20, fill="x")

if __name__ == "__main__":
    app = SoftRobotApp()
    app.mainloop()
