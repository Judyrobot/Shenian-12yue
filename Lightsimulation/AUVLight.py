import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from matplotlib import cm
import tkinter as tk
from tkinter import ttk, messagebox, font, filedialog
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")  # Set matplotlib backend to TkAgg

class UnderwaterLightingSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Underwater Lighting Simulator")
        self.root.geometry("1600x1000")
        
        # Ensure proper font rendering
        plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
        
        # Initialize parameters
        self.initialize_parameters()
        
        # Initialize collapsible panel state variables
        self.expand_vars = {}
        
        # Create UI widgets
        self.create_widgets()
        
    def initialize_parameters(self):
        # Coordinate system parameters - now user-configurable
        self.x_min, self.x_max = -8, 8
        self.y_min, self.y_max = -8, 8
        self.z_min, self.z_max = -2, 60
        
        # Water depth parameter
        self.water_depth = 10  # Unit: m
        
        # Volume calculation range
        self.max_z_for_volume = 60  
        
        # Light source parameters - using custom wavelength names
        self.lamps = [
            {
                'position': (1, 0, 0),
                'I0_base': 30.0,
                'alpha': np.deg2rad(2.5),  # Scattering angle
                'beta': np.deg2rad(0),     # Offset angle
                'wavelength': "532nm(Green)"  # Using wavelength as name
            },
            {
                'position': (0.5, np.sqrt(3)/2, 0),
                'I0_base': 10.0,
                'alpha': np.deg2rad(20),
                'beta': np.deg2rad(0),
                'wavelength': "450nm(Blue)"
            },
            {
                'position': (-0.5, np.sqrt(3)/2, 0),
                'I0_base': 10.0,
                'alpha': np.deg2rad(20),
                'beta': np.deg2rad(0),
                'wavelength': "450nm(Blue)"
            },
            {
                'position': (-1, 0, 0),
                'I0_base': 10.0,
                'alpha': np.deg2rad(20),
                'beta': np.deg2rad(0),
                'wavelength': "450nm(Blue)"
            },
            {
                'position': (-0.5, -np.sqrt(3)/2, 0),
                'I0_base': 10.0,
                'alpha': np.deg2rad(20),
                'beta': np.deg2rad(0),
                'wavelength': "450nm(Blue)"
            },
            {
                'position': (0.5, -np.sqrt(3)/2, 0),
                'I0_base': 10.0,
                'alpha': np.deg2rad(20),
                'beta': np.deg2rad(0),
                'wavelength': "450nm(Blue)"
            }
        ]
        
        # Custom wavelength optical parameters - key: wavelength name, value: absorption and scattering coefficients
        self.wavelength_params = {
            "650nm(Red)": {"absorb": 0.35, "scatter": 0.12},
            "532nm(Green)": {"absorb": 0.20, "scatter": 0.08},
            "450nm(Blue)": {"absorb": 0.18, "scatter": 0.10}
        }
        
        # Distance attenuation parameters
        self.d0 = 1  # Reference distance (unit: m)
        self.distance_decay_factor = 1.2  # Attenuation coefficient (n-th power)
        
        # Scattering and turbulence parameters
        self.g = 0.95  # Scattering anisotropy factor (asymmetry factor)
        self.turbulence_strength = 0.01  # Turbulence intensity
        self.turbulence_correlation = 1.0  # Turbulence spatial correlation
        
        # Detection threshold
        self.I_th = 0.02  
        
        # New: Minimum number of lights required for valid coverage
        self.min_required_lamps = 4  # Default value
        
        # Grid sampling density parameters
        self.x_samples = 60
        self.y_samples = 60
        self.z_samples = 50
        
        # Generate grid
        self.update_grid()
        
        # Beam model parameters
        self.beam_divergence_factor = 0.01  # Beam divergence factor
        self.max_beam_angle = np.pi/3  # Maximum beam angle (radians)
        self.angle_attenuation_factor = 0.5  # Angle attenuation factor
        
        # Particle concentration parameters
        self.base_particle_concentration = 0.05  # Base particle concentration
        self.depth_particle_factor = 0.008  # Depth impact factor on particle concentration
        
        # Absorption and scattering enhancement factors
        self.absorb_enhance_factor = 0.3  # Absorption enhancement factor
        self.scatter_enhance_factor = 1.0  # Scattering enhancement factor
        
        # Background noise factor
        self.background_noise_factor = 0.02  # Background noise factor
        
        # Calculation optimization parameters
        self.light_range_threshold = 4  # Beam range judgment threshold
        self.light_saturation_distance = 0.1  # Light intensity saturation distance near source
        
        # Result variables
        self.valid_points = np.array([])
        self.valid_intensities = np.array([])
        self.valid_coverage_counts = np.array([])  # New: Record how many light sources cover each point
        self.valid_points_for_volume = np.array([])
        self.volume = 0
        self.max_xy_area = 0
        self.xoz_area = 0
        self.yoz_area = 0
        self.nearest_point = None
        self.farthest_point = None
        self.max_xy_points = None
        self.max_xy_z = 0
        
        # Currently selected light source index and wavelength index
        self.selected_lamp_index = 0
        self.selected_wavelength = "532nm(Green)"  # Default selected wavelength
    
    def update_grid(self):
        """Update sampling grid based on current coordinate range and sample counts"""
        self.x_range = np.linspace(self.x_min, self.x_max, self.x_samples)
        self.y_range = np.linspace(self.y_min, self.y_max, self.y_samples)
        self.z_range = np.linspace(self.z_min, self.z_max, self.z_samples)
    
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left parameter panel (collapsible)
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create right result display panel
        result_frame = ttk.LabelFrame(main_frame, text="Result Display", padding="10")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ---------------------- Parameter collapsible panels ----------------------
        # Create parameter category collapsible panels
        self.param_panels = {}
        
        # 1. Coordinate system panel (new)
        self.param_panels['coordinate_system'] = self.create_collapsible_panel(
            left_frame, "Coordinate System", self.create_coordinate_system_panel)
        
        # 2. Light source parameters panel
        self.param_panels['light_source'] = self.create_collapsible_panel(
            left_frame, "Light Source Params", self.create_light_source_panel)
        
        # 3. Water properties panel
        self.param_panels['water_properties'] = self.create_collapsible_panel(
            left_frame, "Water Properties", self.create_water_properties_panel)
        
        # 4. Beam properties panel
        self.param_panels['beam_properties'] = self.create_collapsible_panel(
            left_frame, "Beam Properties", self.create_beam_properties_panel)
        
        # 5. Environmental effects panel
        self.param_panels['environmental'] = self.create_collapsible_panel(
            left_frame, "Environmental Effects", self.create_environmental_panel)
        
        # 6. Simulation parameters panel
        self.param_panels['simulation'] = self.create_collapsible_panel(
            left_frame, "Simulation Params", self.create_simulation_panel)
        
        # 7. Coverage parameters panel
        self.param_panels['coverage'] = self.create_collapsible_panel(
            left_frame, "Coverage Params", self.create_coverage_panel)
        
        # 8. Custom wavelength panel
        self.param_panels['wavelengths'] = self.create_collapsible_panel(
            left_frame, "Custom Wavelengths", self.create_wavelength_panel)
        
        # Run simulation button (fixed at left bottom)
        ttk.Button(left_frame, text="Run Simulation", command=self.run_simulation, style='Accent.TButton').pack(
            side=tk.BOTTOM, pady=15, fill=tk.X)
        
        # ---------------------- Result display area ----------------------
        # Add save and view buttons area
        button_frame = ttk.Frame(result_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Save buttons
        ttk.Button(button_frame, text="Save Combined Image", command=self.save_figure, style='Accent.TButton').pack(
            side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Data", command=self.save_data, style='Accent.TButton').pack(
            side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # Individual plot view buttons
        ttk.Button(button_frame, text="View 3D Plot", command=lambda: self.show_individual_plot(1), style='TButton').pack(
            side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View XY Section", command=lambda: self.show_individual_plot(2), style='TButton').pack(
            side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View XOZ Projection", command=lambda: self.show_individual_plot(3), style='TButton').pack(
            side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View YOZ Projection", command=lambda: self.show_individual_plot(4), style='TButton').pack(
            side=tk.LEFT, padx=5)
        
        # Result text area
        self.result_text = tk.Text(result_frame, height=10, width=60)
        self.result_text.pack(fill=tk.X, pady=5)
        self.result_text.config(state=tk.DISABLED)
        
        # Graph display area
        self.fig = plt.figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize styles
        self.setup_styles()
    
    def setup_styles(self):
        """Set up UI styles"""
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        style.configure('Section.TLabel', font=('Arial', 10, 'bold'))
        style.configure('Collapsible.TFrame', background='#f0f0f0')
    
    def create_collapsible_panel(self, parent, title, content_creator):
        """Create collapsible panel"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        
        # Header bar
        header = ttk.Frame(frame)
        header.pack(fill=tk.X)
        
        # Use class-level expand_vars dictionary
        self.expand_vars[title] = tk.BooleanVar(value=False)
        
        # Expand/collapse button
        btn = ttk.Checkbutton(
            header, 
            variable=self.expand_vars[title],
            text=title,
            style='Section.TLabel',
            command=lambda: self.toggle_panel(content, self.expand_vars[title])
        )
        btn.pack(side=tk.LEFT)
        
        # Content area
        content = ttk.Frame(frame, style='Collapsible.TFrame')
        
        # Create content
        content_creator(content)
        
        # Collapse by default
        self.toggle_panel(content, self.expand_vars[title])
        
        return frame
    
    def toggle_panel(self, content, var):
        """Toggle panel expand/collapse state"""
        if var.get():
            content.pack(fill=tk.X, padx=5, pady=5)
        else:
            content.pack_forget()
    
    # New: Create coordinate system panel
    def create_coordinate_system_panel(self, parent):
        """Create coordinate system parameter panel content"""
        # X-axis range
        ttk.Label(parent, text="X-axis Range (m):").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        x_frame = ttk.Frame(parent)
        x_frame.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        self.x_min_var = tk.StringVar(value=str(self.x_min))
        self.x_max_var = tk.StringVar(value=str(self.x_max))
        
        ttk.Label(x_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Entry(x_frame, textvariable=self.x_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(x_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(x_frame, textvariable=self.x_max_var, width=8).pack(side=tk.LEFT, padx=2)
        
        # Y-axis range
        ttk.Label(parent, text="Y-axis Range (m):").grid(row=1, column=0, sticky=tk.W, pady=2)
        
        y_frame = ttk.Frame(parent)
        y_frame.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        self.y_min_var = tk.StringVar(value=str(self.y_min))
        self.y_max_var = tk.StringVar(value=str(self.y_max))
        
        ttk.Label(y_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Entry(y_frame, textvariable=self.y_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(y_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(y_frame, textvariable=self.y_max_var, width=8).pack(side=tk.LEFT, padx=2)
        
        # Z-axis range
        ttk.Label(parent, text="Z-axis Range (m):").grid(row=2, column=0, sticky=tk.W, pady=2)
        
        z_frame = ttk.Frame(parent)
        z_frame.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        self.z_min_var = tk.StringVar(value=str(self.z_min))
        self.z_max_var = tk.StringVar(value=str(self.z_max))
        
        ttk.Label(z_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Entry(z_frame, textvariable=self.z_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(z_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(z_frame, textvariable=self.z_max_var, width=8).pack(side=tk.LEFT, padx=2)
        
        # Volume calculation range
        ttk.Label(parent, text="Max Z for Volume (m):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.max_z_for_volume_var = tk.StringVar(value=str(self.max_z_for_volume))
        ttk.Entry(parent, textvariable=self.max_z_for_volume_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Save button
        ttk.Button(parent, text="Save Coordinate Params", command=self.save_coordinate_parameters).grid(
            row=4, column=0, columnspan=2, pady=10)
        
        # Hint text
        ttk.Label(
            parent, 
            text="Note: Adjust coordinate ranges to focus on specific areas\nAffects both simulation and visualization", 
            font=('Arial', 8),
            foreground='gray'
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)
    
    # New: Save coordinate system parameters
    def save_coordinate_parameters(self):
        """Save coordinate system parameters"""
        try:
            x_min = float(self.x_min_var.get())
            x_max = float(self.x_max_var.get())
            y_min = float(self.y_min_var.get())
            y_max = float(self.y_max_var.get())
            z_min = float(self.z_min_var.get())
            z_max = float(self.z_max_var.get())
            max_z_vol = float(self.max_z_for_volume_var.get())
            
            # Validate ranges
            if x_min >= x_max:
                raise ValueError("X min must be less than X max")
            if y_min >= y_max:
                raise ValueError("Y min must be less than Y max")
            if z_min >= z_max:
                raise ValueError("Z min must be less than Z max")
            if max_z_vol < z_min or max_z_vol > z_max:
                raise ValueError("Max Z for Volume must be within Z range")
            
            # Update parameters
            self.x_min, self.x_max = x_min, x_max
            self.y_min, self.y_max = y_min, y_max
            self.z_min, self.z_max = z_min, z_max
            self.max_z_for_volume = max_z_vol
            
            # Update grid with new ranges
            self.update_grid()
            
            messagebox.showinfo("Success", "Coordinate system parameters saved")
        except ValueError as e:
            messagebox.showerror("Error", f"Coordinate parameter input error: {str(e)}")
    
    def create_light_source_panel(self, parent):
        """Create light source parameter panel content"""
        # Light source selection
        ttk.Label(parent, text="Select Light Source:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.lamp_combobox = ttk.Combobox(parent, values=[f"Light {i+1}" for i in range(len(self.lamps))], state="readonly")
        self.lamp_combobox.current(0)
        self.lamp_combobox.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.lamp_combobox.bind("<<ComboboxSelected>>", self.on_lamp_selected)
        
        # Light source position parameters
        ttk.Label(parent, text="X Position (m):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.lamp_x = tk.StringVar(value=str(self.lamps[0]['position'][0]))
        ttk.Entry(parent, textvariable=self.lamp_x, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Y Position (m):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.lamp_y = tk.StringVar(value=str(self.lamps[0]['position'][1]))
        ttk.Entry(parent, textvariable=self.lamp_y, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Z Position (m):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.lamp_z = tk.StringVar(value=str(self.lamps[0]['position'][2]))
        ttk.Entry(parent, textvariable=self.lamp_z, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Light source optical parameters
        ttk.Label(parent, text="Base Intensity:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.lamp_I0 = tk.StringVar(value=str(self.lamps[0]['I0_base']))
        ttk.Entry(parent, textvariable=self.lamp_I0, width=10).grid(row=4, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Scatter Angle (°):").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.lamp_alpha = tk.StringVar(value=str(np.rad2deg(self.lamps[0]['alpha'])))
        ttk.Entry(parent, textvariable=self.lamp_alpha, width=10).grid(row=5, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Offset Angle (°):").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.lamp_beta = tk.StringVar(value=str(np.rad2deg(self.lamps[0]['beta'])))
        ttk.Entry(parent, textvariable=self.lamp_beta, width=10).grid(row=6, column=1, sticky=tk.W, pady=2)
        
        # Wavelength selection (using custom wavelength list)
        ttk.Label(parent, text="Wavelength:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.lamp_wavelength = ttk.Combobox(parent, values=list(self.wavelength_params.keys()), state="readonly")
        self.lamp_wavelength.current(list(self.wavelength_params.keys()).index(self.lamps[0]['wavelength']))
        self.lamp_wavelength.grid(row=7, column=1, sticky=tk.W, pady=2)
        
        # Light source operation buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=8, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Save Params", command=self.save_lamp_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Add Light", command=self.add_lamp).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Light", command=self.delete_lamp).pack(side=tk.LEFT, padx=5)
    
    def create_water_properties_panel(self, parent):
        """Create water properties panel content"""
        # Basic water parameters
        ttk.Label(parent, text="Water Depth (m):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.water_depth_var = tk.StringVar(value=str(self.water_depth))
        ttk.Entry(parent, textvariable=self.water_depth_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # Particle concentration parameters
        ttk.Label(parent, text="Base Particle Conc.:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.base_particle_var = tk.StringVar(value=str(self.base_particle_concentration))
        ttk.Entry(parent, textvariable=self.base_particle_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Depth Particle Factor:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.depth_particle_var = tk.StringVar(value=str(self.depth_particle_factor))
        ttk.Entry(parent, textvariable=self.depth_particle_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Save button
        ttk.Button(parent, text="Save Water Params", command=self.save_water_parameters).grid(row=3, column=0, columnspan=2, pady=10)
    
    def create_beam_properties_panel(self, parent):
        """Create beam properties panel content"""
        # Beam angle parameters
        ttk.Label(parent, text="Max Beam Angle (°):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_beam_angle_var = tk.StringVar(value=str(np.rad2deg(self.max_beam_angle)))
        ttk.Entry(parent, textvariable=self.max_beam_angle_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Angle Attenuation Factor:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.angle_attenuation_var = tk.StringVar(value=str(self.angle_attenuation_factor))
        ttk.Entry(parent, textvariable=self.angle_attenuation_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Beam divergence parameters
        ttk.Label(parent, text="Beam Divergence Factor:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.beam_divergence_var = tk.StringVar(value=str(self.beam_divergence_factor))
        ttk.Entry(parent, textvariable=self.beam_divergence_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Absorption and scattering enhancement
        ttk.Label(parent, text="Absorption Enhance Factor:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.absorb_enhance_var = tk.StringVar(value=str(self.absorb_enhance_factor))
        ttk.Entry(parent, textvariable=self.absorb_enhance_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Scattering Enhance Factor:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.scatter_enhance_var = tk.StringVar(value=str(self.scatter_enhance_factor))
        ttk.Entry(parent, textvariable=self.scatter_enhance_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=2)
        
        # Save button
        ttk.Button(parent, text="Save Beam Params", command=self.save_beam_parameters).grid(row=5, column=0, columnspan=2, pady=10)
    
    def create_environmental_panel(self, parent):
        """Create environmental effects panel content"""
        # Turbulence parameters
        ttk.Label(parent, text="Turbulence Strength:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.turbulence_strength_var = tk.StringVar(value=str(self.turbulence_strength))
        ttk.Entry(parent, textvariable=self.turbulence_strength_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Turbulence Correlation:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.turbulence_correlation_var = tk.StringVar(value=str(self.turbulence_correlation))
        ttk.Entry(parent, textvariable=self.turbulence_correlation_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Scattering parameters
        ttk.Label(parent, text="Scattering Asymmetry (g):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.asymmetry_factor = tk.StringVar(value=str(self.g))
        ttk.Entry(parent, textvariable=self.asymmetry_factor, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Noise parameters
        ttk.Label(parent, text="Background Noise Factor:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.background_noise_var = tk.StringVar(value=str(self.background_noise_factor))
        ttk.Entry(parent, textvariable=self.background_noise_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Detection threshold
        ttk.Label(parent, text="Detection Threshold:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.I_th_var = tk.StringVar(value=str(self.I_th))
        ttk.Entry(parent, textvariable=self.I_th_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=2)
        
        # Save button
        ttk.Button(parent, text="Save Env Params", command=self.save_environmental_parameters).grid(row=5, column=0, columnspan=2, pady=10)
    
    def create_simulation_panel(self, parent):
        """Create simulation parameters panel content"""
        # Distance attenuation parameters
        ttk.Label(parent, text="Ref Distance (d0, m):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ref_distance = tk.StringVar(value=str(self.d0))
        ttk.Entry(parent, textvariable=self.ref_distance, width=10).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Distance Decay Coeff (n):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.distance_factor = tk.StringVar(value=str(self.distance_decay_factor))
        ttk.Entry(parent, textvariable=self.distance_factor, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Sampling parameters
        ttk.Label(parent, text="X Samples:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.x_samples_var = tk.StringVar(value=str(self.x_samples))
        ttk.Entry(parent, textvariable=self.x_samples_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Y Samples:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.y_samples_var = tk.StringVar(value=str(self.y_samples))
        ttk.Entry(parent, textvariable=self.y_samples_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Z Samples:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.z_samples_var = tk.StringVar(value=str(self.z_samples))
        ttk.Entry(parent, textvariable=self.z_samples_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=2)
        
        # Calculation optimization parameters
        ttk.Label(parent, text="Beam Range Threshold (m):").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.light_range_var = tk.StringVar(value=str(self.light_range_threshold))
        ttk.Entry(parent, textvariable=self.light_range_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Light Saturation Distance (m):").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.light_saturation_var = tk.StringVar(value=str(self.light_saturation_distance))
        ttk.Entry(parent, textvariable=self.light_saturation_var, width=10).grid(row=6, column=1, sticky=tk.W, pady=2)
        
        # Save button
        ttk.Button(parent, text="Save Sim Params", command=self.save_simulation_parameters).grid(row=7, column=0, columnspan=2, pady=10)
    
    def create_coverage_panel(self, parent):
        """Create coverage parameters panel content"""
        ttk.Label(parent, text="Minimum Required Lights:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Create options from 1 to total number of light sources
        max_lamp = len(self.lamps)
        self.min_required_lamps_var = tk.IntVar(value=self.min_required_lamps)
        
        # Create radio button group
        frame = ttk.Frame(parent)
        frame.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        for i in range(1, max_lamp + 1):
            ttk.Radiobutton(
                frame, 
                text=str(i), 
                variable=self.min_required_lamps_var, 
                value=i
            ).pack(side=tk.LEFT, padx=10)
        
        # Add description text
        ttk.Label(
            parent, 
            text="Set how many lights must cover a point to be valid\nControls reliability of illuminated space", 
            font=('Arial', 8),
            foreground='gray'
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Save button
        ttk.Button(parent, text="Save Coverage Params", command=self.save_coverage_parameters).grid(row=3, column=0, columnspan=2, pady=10)
    
    def create_wavelength_panel(self, parent):
        """Create custom wavelength parameter panel"""
        # Wavelength selection
        ttk.Label(parent, text="Select Wavelength:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.wavelength_combobox = ttk.Combobox(parent, values=list(self.wavelength_params.keys()), state="readonly")
        self.wavelength_combobox.current(0)
        self.wavelength_combobox.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.wavelength_combobox.bind("<<ComboboxSelected>>", self.on_wavelength_selected)
        
        # Wavelength name (allow user-defined names)
        ttk.Label(parent, text="Wavelength Name:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.wavelength_name = tk.StringVar(value=list(self.wavelength_params.keys())[0])
        ttk.Entry(parent, textvariable=self.wavelength_name, width=15).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Optical parameters
        ttk.Label(parent, text="Absorption Coeff:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.wavelength_absorb = tk.StringVar(value=str(self.wavelength_params[list(self.wavelength_params.keys())[0]]["absorb"]))
        ttk.Entry(parent, textvariable=self.wavelength_absorb, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(parent, text="Scattering Coeff:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.wavelength_scatter = tk.StringVar(value=str(self.wavelength_params[list(self.wavelength_params.keys())[0]]["scatter"]))
        ttk.Entry(parent, textvariable=self.wavelength_scatter, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Wavelength operation buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Save Wavelength", command=self.save_wavelength_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Add Wavelength", command=self.add_wavelength).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Wavelength", command=self.delete_wavelength).pack(side=tk.LEFT, padx=5)
        
        # Hint information
        ttk.Label(parent, text="Hint: Name can include wavelength and color, e.g.'589nm(Yellow)'", 
                 font=('Arial', 8)).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)
    
    def on_wavelength_selected(self, event):
        """Update parameter display when different wavelength is selected"""
        wavelength = self.wavelength_combobox.get()
        self.selected_wavelength = wavelength
        params = self.wavelength_params[wavelength]
        
        self.wavelength_name.set(wavelength)
        self.wavelength_absorb.set(str(params["absorb"]))
        self.wavelength_scatter.set(str(params["scatter"]))
        
        # Update wavelength selection dropdown in light source panel
        self.lamp_wavelength['values'] = list(self.wavelength_params.keys())
    
    def save_wavelength_parameters(self):
        """Save current wavelength parameters"""
        try:
            old_name = self.selected_wavelength
            new_name = self.wavelength_name.get().strip()
            
            if not new_name:
                messagebox.showerror("Error", "Wavelength name cannot be empty")
                return
                
            # If name changed, delete old name first
            if new_name != old_name and old_name in self.wavelength_params:
                del self.wavelength_params[old_name]
            
            # Save new parameters
            self.wavelength_params[new_name] = {
                "absorb": float(self.wavelength_absorb.get()),
                "scatter": float(self.wavelength_scatter.get())
            }
            
            # Update selected wavelength name
            self.selected_wavelength = new_name
            
            # Update dropdown
            self.wavelength_combobox['values'] = list(self.wavelength_params.keys())
            self.wavelength_combobox.current(list(self.wavelength_params.keys()).index(new_name))
            
            # Also update wavelength selection dropdown in light source panel
            self.lamp_wavelength['values'] = list(self.wavelength_params.keys())
            
            messagebox.showinfo("Success", f"Wavelength '{new_name}' parameters saved")
        except ValueError as e:
            messagebox.showerror("Error", f"Parameter input error: {str(e)}")
    
    def add_wavelength(self):
        """Add new custom wavelength"""
        # Generate default name (e.g."NewWavelength1", "NewWavelength2", etc.)
        default_name = "NewWavelength1"
        counter = 2
        while default_name in self.wavelength_params:
            default_name = f"NewWavelength{counter}"
            counter += 1
        
        # Default parameters (between red and blue light)
        self.wavelength_params[default_name] = {
            "absorb": 0.25,
            "scatter": 0.10
        }
        
        # Update dropdown
        self.wavelength_combobox['values'] = list(self.wavelength_params.keys())
        self.wavelength_combobox.current(list(self.wavelength_params.keys()).index(default_name))
        self.on_wavelength_selected(None)
        
        messagebox.showinfo("Success", f"Added new wavelength '{default_name}'")
    
    def delete_wavelength(self):
        """Delete currently selected wavelength"""
        if len(self.wavelength_params) <= 1:
            messagebox.showwarning("Warning", "Keep at least one wavelength")
            return
            
        wavelength = self.selected_wavelength
        
        # Check if any light sources are using this wavelength
        using_lamps = [i+1 for i, lamp in enumerate(self.lamps) if lamp['wavelength'] == wavelength]
        if using_lamps:
            response = messagebox.askyesno("Confirm", 
                                         f"Wavelength '{wavelength}' is used by lights {using_lamps},\nThese lights will use the first wavelength after deletion.\nContinue?")
            if not response:
                return
        
        # Delete wavelength
        del self.wavelength_params[wavelength]
        
        # Update dropdown
        wavelengths = list(self.wavelength_params.keys())
        self.wavelength_combobox['values'] = wavelengths
        self.wavelength_combobox.current(0)
        self.on_wavelength_selected(None)
        
        # Update light sources using this wavelength
        first_wavelength = wavelengths[0] if wavelengths else ""
        for i in range(len(self.lamps)):
            if self.lamps[i]['wavelength'] == wavelength:
                self.lamps[i]['wavelength'] = first_wavelength
        
        # Update wavelength selection in light source panel
        if self.lamp_combobox.current() >= 0:
            self.on_lamp_selected(None)
        
        messagebox.showinfo("Success", f"Deleted wavelength '{wavelength}'")
    
    def on_lamp_selected(self, event):
        # Update parameter display when different light source is selected
        index = self.lamp_combobox.current()
        self.selected_lamp_index = index
        lamp = self.lamps[index]
        
        self.lamp_x.set(str(lamp['position'][0]))
        self.lamp_y.set(str(lamp['position'][1]))
        self.lamp_z.set(str(lamp['position'][2]))
        self.lamp_I0.set(str(lamp['I0_base']))
        self.lamp_alpha.set(str(np.rad2deg(lamp['alpha'])))
        self.lamp_beta.set(str(np.rad2deg(lamp['beta'])))
        
        # Update wavelength selection dropdown (ensure values exist)
        wavelength_values = list(self.wavelength_params.keys())
        self.lamp_wavelength['values'] = wavelength_values
        
        # If current light's wavelength not in list, use first wavelength
        if lamp['wavelength'] in wavelength_values:
            self.lamp_wavelength.current(wavelength_values.index(lamp['wavelength']))
        else:
            self.lamp_wavelength.current(0)
    
    def save_lamp_parameters(self):
        # Save current light source parameters
        try:
            index = self.selected_lamp_index
            self.lamps[index] = {
                'position': (float(self.lamp_x.get()), float(self.lamp_y.get()), float(self.lamp_z.get())),
                'I0_base': float(self.lamp_I0.get()),
                'alpha': np.deg2rad(float(self.lamp_alpha.get())),
                'beta': np.deg2rad(float(self.lamp_beta.get())),
                'wavelength': self.lamp_wavelength.get()
            }
            messagebox.showinfo("Success", f"Light {index+1} parameters saved")
        except ValueError as e:
            messagebox.showerror("Error", f"Parameter input error: {str(e)}")
    
    def add_lamp(self):
        # Add new light source
        new_index = len(self.lamps)
        # Default parameters, using first wavelength
        first_wavelength = list(self.wavelength_params.keys())[0] if self.wavelength_params else ""
        new_lamp = {
            'position': (0, 0, 0),
            'I0_base': 10.0,
            'alpha': np.deg2rad(20),
            'beta': np.deg2rad(0),
            'wavelength': first_wavelength
        }
        self.lamps.append(new_lamp)
        
        # Update dropdown
        self.lamp_combobox['values'] = [f"Light {i+1}" for i in range(len(self.lamps))]
        self.lamp_combobox.current(new_index)
        self.on_lamp_selected(None)
        
        # Update coverage parameter panel options
        self.update_coverage_panel_options()
        
        messagebox.showinfo("Success", f"Added new light {new_index+1}")
    
    def delete_lamp(self):
        # Delete current light source
        if len(self.lamps) <= 1:
            messagebox.showwarning("Warning", "Keep at least one light source")
            return
            
        index = self.selected_lamp_index
        del self.lamps[index]
        
        # Update dropdown
        self.lamp_combobox['values'] = [f"Light {i+1}" for i in range(len(self.lamps))]
        new_index = min(index, len(self.lamps)-1)
        self.lamp_combobox.current(new_index)
        self.on_lamp_selected(None)
        
        # Update coverage parameter panel options
        self.update_coverage_panel_options()
        
        # If current minimum required lights exceeds remaining count, adjust automatically
        if self.min_required_lamps > len(self.lamps):
            self.min_required_lamps = len(self.lamps)
            self.min_required_lamps_var.set(self.min_required_lamps)
            messagebox.showinfo("Note", f"Minimum required lights adjusted to {self.min_required_lamps} (current total)")
        
        messagebox.showinfo("Success", f"Deleted light {index+1}")
    
    def update_coverage_panel_options(self):
        """Update coverage parameter panel options when light count changes"""
        # Find coverage parameter panel
        for child in self.param_panels['coverage'].winfo_children():
            for subchild in child.winfo_children():
                if isinstance(subchild, ttk.Frame) and subchild.winfo_children():
                    # Clear existing radio buttons
                    for btn in subchild.winfo_children():
                        btn.destroy()
                    
                    # Create new radio button group
                    max_lamp = len(self.lamps)
                    for i in range(1, max_lamp + 1):
                        ttk.Radiobutton(
                            subchild, 
                            text=str(i), 
                            variable=self.min_required_lamps_var, 
                            value=i
                        ).pack(side=tk.LEFT, padx=10)
                    break
    
    def save_water_parameters(self):
        """Save water property settings"""
        try:
            self.water_depth = float(self.water_depth_var.get())
            self.base_particle_concentration = float(self.base_particle_var.get())
            self.depth_particle_factor = float(self.depth_particle_var.get())
            
            messagebox.showinfo("Success", "Water parameters saved")
        except ValueError as e:
            messagebox.showerror("Error", f"Water parameter input error: {str(e)}")
    
    def save_beam_parameters(self):
        """Save beam model parameters"""
        try:
            self.beam_divergence_factor = float(self.beam_divergence_var.get())
            self.max_beam_angle = np.deg2rad(float(self.max_beam_angle_var.get()))
            self.angle_attenuation_factor = float(self.angle_attenuation_var.get())
            self.absorb_enhance_factor = float(self.absorb_enhance_var.get())
            self.scatter_enhance_factor = float(self.scatter_enhance_var.get())
            
            messagebox.showinfo("Success", "Beam parameters saved")
        except ValueError as e:
            messagebox.showerror("Error", f"Beam parameter input error: {str(e)}")
    
    def save_environmental_parameters(self):
        """Save environmental effect parameters"""
        try:
            self.turbulence_strength = float(self.turbulence_strength_var.get())
            self.turbulence_correlation = float(self.turbulence_correlation_var.get())
            self.g = float(self.asymmetry_factor.get())
            self.background_noise_factor = float(self.background_noise_var.get())
            self.I_th = float(self.I_th_var.get())
            
            messagebox.showinfo("Success", "Environmental parameters saved")
        except ValueError as e:
            messagebox.showerror("Error", f"Environmental parameter input error: {str(e)}")
    
    def save_simulation_parameters(self):
        """Save simulation parameter settings"""
        try:
            self.d0 = float(self.ref_distance.get())
            self.distance_decay_factor = float(self.distance_factor.get())
            self.x_samples = int(self.x_samples_var.get())
            self.y_samples = int(self.y_samples_var.get())
            self.z_samples = int(self.z_samples_var.get())
            self.light_range_threshold = float(self.light_range_var.get())
            self.light_saturation_distance = float(self.light_saturation_var.get())
            
            # Update grid
            self.update_grid()
            
            messagebox.showinfo("Success", "Simulation parameters saved")
        except ValueError as e:
            messagebox.showerror("Error", f"Simulation parameter input error: {str(e)}")
    
    def save_coverage_parameters(self):
        """Save coverage parameter settings"""
        try:
            self.min_required_lamps = self.min_required_lamps_var.get()
            
            # Check if exceeds current total light sources
            if self.min_required_lamps > len(self.lamps):
                self.min_required_lamps = len(self.lamps)
                self.min_required_lamps_var.set(self.min_required_lamps)
                messagebox.showinfo("Note", f"Minimum required lights adjusted to {self.min_required_lamps} (current total)")
                return
                
            messagebox.showinfo("Success", f"Coverage parameters saved, minimum {self.min_required_lamps} lights required")
        except ValueError as e:
            messagebox.showerror("Error", f"Coverage parameter input error: {str(e)}")
    
    def save_figure(self, fig=None, default_filename="underwater_lighting_simulation_result.png"):
        """Save current figure to file"""
        if fig is None:
            fig = self.fig
            
        if fig is None:
            messagebox.showwarning("Warning", "No figure to save, run simulation first")
            return
            
        # Open file dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("SVG Vector", "*.svg"),
                ("PDF Document", "*.pdf"),
                ("All Files", "*.*")
            ],
            initialfile=default_filename,
            title="Save Figure"
        )
        
        if file_path:
            try:
                # Save figure
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Figure saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save figure:\n{str(e)}")
    
    def save_data(self):
        """Save simulation data results"""
        if len(self.valid_points) == 0:
            messagebox.showwarning("Warning", "No data to save, run simulation first")
            return
            
        # Default filename
        default_filename = "underwater_lighting_simulation_data.csv"
        
        # Open file dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV File", "*.csv"),
                ("Text File", "*.txt"),
                ("All Files", "*.*")
            ],
            initialfile=default_filename,
            title="Save Data"
        )
        
        if file_path:
            try:
                # Create data dictionary, including how many light sources cover each point
                data = {
                    'X Coordinate(m)': self.valid_points[:, 0],
                    'Y Coordinate(m)': self.valid_points[:, 1],
                    'Z Coordinate(m)': self.valid_points[:, 2],
                    'Intensity': self.valid_intensities,
                    'Covering Lights': self.valid_coverage_counts
                }
                
                # Create DataFrame and save
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                
                # Also save result summary
                summary_path = os.path.splitext(file_path)[0] + "_summary.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(self.result_text.get(1.0, tk.END))
                
                messagebox.showinfo("Success", f"Data saved to:\n{file_path}\nSummary saved to:\n{summary_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data:\n{str(e)}")
    
    def particle_concentration(self, water_depth):
        """Simulate suspended particle concentration in water (unit: kg/m³)"""
        return self.base_particle_concentration + self.depth_particle_factor * water_depth
    
    def henry_greenstein(self, theta, g):
        """Henry-Greenstein scattering phase function"""
        return (1 - g**2) / (1 + g**2 - 2*g*np.cos(theta))**1.5
    
    def get_turbulence_perturbation(self, x, y):
        """Simulate water turbulence perturbation on beam direction"""
        np.random.seed(42)
        noise = np.random.randn(100, 100)
        smoothed_noise = gaussian_filter(noise, sigma=self.turbulence_correlation)
        i = int((x - self.x_min)/(self.x_max - self.x_min) * 99)
        j = int((y - self.y_min)/(self.y_max - self.y_min) * 99)
        i = np.clip(i, 0, 99)
        j = np.clip(j, 0, 99)
        return smoothed_noise[i, j] * self.turbulence_strength
    
    def calculate_light_intensity(self, point, lamp_params):
        """Calculate light intensity at a point from a specific light source"""
        x, y, z = point
        lamp_x, lamp_y, lamp_z = lamp_params['position']
        
        # Calculate vector and distance from light source to target point
        vec = np.array([x - lamp_x, y - lamp_y, z - lamp_z])
        d = np.linalg.norm(vec)
        if d < self.light_saturation_distance:  # Light intensity saturation near source
            return lamp_params['I0_base']
        
        # Turbulence perturbation
        turb_perturb = self.get_turbulence_perturbation(x, y)
        beta_noisy = lamp_params['beta'] + turb_perturb
        
        # Calculate light source angle in XOY plane (for direction vector)
        dx = lamp_x if lamp_x != 0 else 1e-10  # Avoid division by zero
        lamp_theta = np.arctan2(lamp_y, dx)
        theta_noisy = lamp_theta + turb_perturb
        
        # Beam central axis direction vector
        axis_vec_z = np.array([0, 0, np.cos(beta_noisy)])
        axis_vec_xy = np.array([np.sin(beta_noisy)*np.cos(theta_noisy), 
                               np.sin(beta_noisy)*np.sin(theta_noisy), 0])
        axis_vec = axis_vec_z + axis_vec_xy
        axis_vec = axis_vec / np.linalg.norm(axis_vec)
        
        # Particle concentration
        c = self.particle_concentration(self.water_depth)
        
        # Beam divergence angle
        alpha_scattered = lamp_params['alpha'] + self.beam_divergence_factor * c * d
        alpha_scattered = np.clip(alpha_scattered, 0, self.max_beam_angle)
        
        # Determine if point is within beam range
        theta = np.arccos(np.dot(vec, axis_vec) / d)
        # if theta > alpha_scattered:
        #     return 0  # Outside beam range
        
        # Get attenuation coefficients for current wavelength (using custom wavelength parameters)
        wavelength = lamp_params['wavelength']
        if wavelength not in self.wavelength_params:
            # If wavelength doesn't exist, use first wavelength's parameters
            wavelength = next(iter(self.wavelength_params.keys())) if self.wavelength_params else ""
            
        gamma_absorb = self.wavelength_params[wavelength]["absorb"] if wavelength else 0.2
        gamma_scatter = self.wavelength_params[wavelength]["scatter"] if wavelength else 0.1
        
        # Light intensity attenuation calculation
        gamma_total_absorb = gamma_absorb * (1 + self.absorb_enhance_factor * c)
        gamma_total_scatter = gamma_scatter * (1 + self.scatter_enhance_factor * c)
        
        # Absorption attenuation
        I_absorb = np.exp(-gamma_total_absorb * d)
        # Scattering attenuation
        forward_scatter_ratio = (1 + self.g) / 2
        I_scatter = np.exp(-gamma_total_scatter * d) * forward_scatter_ratio
        
        # Angle attenuation
        sigma = alpha_scattered / 2 * (1 + self.angle_attenuation_factor * c)
        angle_factor = np.exp(-0.5 * (theta / sigma) **2)
            
        # Distance attenuation (using user-set attenuation coefficient)
        distance_factor = 1 / (1 + (alpha_scattered * d / self.d0) ** self.distance_decay_factor)
        
        # Background noise
        background_noise = self.background_noise_factor * c
        
        # Total light intensity calculation
        solid_angle = 2 * np.pi * (1 - np.cos(lamp_params['alpha']))
        I0 = lamp_params['I0_base'] / solid_angle
        I = I0 * I_absorb * I_scatter * angle_factor * distance_factor - background_noise
        return max(0, I)
    
    def run_simulation(self):
        # Ensure all parameters are updated
        try:
            self.save_water_parameters()
            self.save_beam_parameters()
            self.save_environmental_parameters()
            self.save_simulation_parameters()
            self.save_coverage_parameters()
            self.save_coordinate_parameters()  # Ensure coordinate parameters are updated
        except ValueError as e:
            messagebox.showerror("Error", f"Parameter input error: {str(e)}")
            return
        
        # Check for available wavelengths
        if not self.wavelength_params:
            messagebox.showerror("Error", "No available wavelength parameters, add wavelength first")
            return
        
        # Show running message
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Running simulation, please wait...\n")
        self.result_text.config(state=tk.DISABLED)
        self.root.update()
        
        # Filter valid observation areas
        valid_points = []  # Valid point coordinates
        valid_intensities = []  # Total light intensity at valid points
        valid_coverage_counts = []  # How many light sources cover each point
        valid_points_for_volume = []  # Valid points for volume calculation
        
        # Extract all light source positions
        lamp_positions = np.array([lamp['position'] for lamp in self.lamps])
        
        for x in self.x_range:
            for y in self.y_range:
                # Preliminary filtering: only consider areas near light sources
                min_dist_xy = np.min([
                    np.sqrt((x - lamp_x)**2 + (y - lamp_y)** 2) 
                    for lamp_x, lamp_y, _ in lamp_positions
                ])
                if min_dist_xy > self.light_range_threshold:  # Light intensity negligible beyond this range
                    continue
                
                for z in self.z_range:
                    # Calculate each light source's contribution to this point and count qualifying sources
                    total_intensity = 0
                    coverage_count = 0  # Count light sources covering this point
                    
                    for lamp in self.lamps:
                        intensity = self.calculate_light_intensity((x, y, z), lamp)
                        total_intensity += intensity
                        if intensity >= self.I_th:  # This light source covers the point
                            coverage_count += 1
                    
                    # Valid condition: total intensity >= threshold and covered by enough light sources
                    if total_intensity >= self.I_th and coverage_count >= self.min_required_lamps:
                        valid_points.append([x, y, z])
                        valid_intensities.append(total_intensity)
                        valid_coverage_counts.append(coverage_count)
                        # If within Z range for volume calculation, record
                        if self.z_min <= z <= self.max_z_for_volume:
                            valid_points_for_volume.append([x, y, z])
        
        # Convert to arrays (handle empty values)
        self.valid_points = np.array(valid_points) if valid_points else np.array([])
        self.valid_points_for_volume = np.array(valid_points_for_volume) if valid_points_for_volume else np.array([])
        self.valid_intensities = np.array(valid_intensities) if valid_intensities else np.array([])
        self.valid_coverage_counts = np.array(valid_coverage_counts) if valid_coverage_counts else np.array([])
        
        # Calculate key metrics
        self.nearest_point = None
        self.farthest_point = None
        self.volume = 0
        self.max_xy_area = 0
        self.xoz_area = 0
        self.yoz_area = 0
        self.max_xy_points = None
        self.max_xy_z = 0
        
        result_text = f"===== Key Metrics (Water Depth {self.water_depth} m, Min {self.min_required_lamps} Lights Required) =====\n"
        
        if len(self.valid_points) > 0:
            # Calculate distance from valid points to nearest light source
            distances = []
            for point in self.valid_points:
                min_dist = np.min(np.linalg.norm(point - lamp_positions, axis=1))
                distances.append(min_dist)
            distances = np.array(distances)
            
            # Nearest and farthest points
            nearest_idx = np.argmin(distances)
            farthest_idx = np.argmax(distances)
            self.nearest_point = self.valid_points[nearest_idx]
            self.farthest_point = self.valid_points[farthest_idx]
            
            result_text += f"Nearest to light source: {self.nearest_point.round(4)}, Intensity: {self.valid_intensities[nearest_idx]:.4f}, Covered by {self.valid_coverage_counts[nearest_idx]} lights\n"
            result_text += f"Farthest from light source: {self.farthest_point.round(4)}, Intensity: {self.valid_intensities[farthest_idx]:.4f}, Covered by {self.valid_coverage_counts[farthest_idx]} lights\n"
            
            # Valid volume
            self.volume = ConvexHull(self.valid_points_for_volume).volume if len(self.valid_points_for_volume) >= 4 else 0
            
            # Maximum cross-sectional area
            z_values = np.unique(self.valid_points[:, 2])
            for z in z_values:
                section = self.valid_points[self.valid_points[:, 2]==z][:, :2]
                if len(section)>=3:
                    try:
                        area = ConvexHull(section).area
                        if area > self.max_xy_area:
                            self.max_xy_area = area
                            self.max_xy_z = z
                            self.max_xy_points = section
                    except:
                        pass
            
            result_text += f"Valid Volume: {self.volume:.2f} m³, Max XOY Section: {self.max_xy_area:.2f} m²\n"
            
            # Projected areas
            self.xoz_area = ConvexHull(self.valid_points[:, [0,2]]).area if len(self.valid_points)>=3 else 0
            self.yoz_area = ConvexHull(self.valid_points[:, [1,2]]).area if len(self.valid_points)>=3 else 0
            
            result_text += f"XOZ Projection Area: {self.xoz_area:.2f} m²\n"
            result_text += f"YOZ Projection Area: {self.yoz_area:.2f} m²\n"
            result_text += f"Detection Threshold: {self.I_th:.4f}\n"
        else:
            result_text += f"No valid area found with total intensity ≥ {self.I_th} and covered by at least {self.min_required_lamps} lights (Water Depth {self.water_depth} m)\n"
        
        # Display results
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)
        
        # Update visualization
        self.update_visualization()
    
    def update_visualization(self):
        # Clear existing figure
        self.fig.clear()
        
        # Figure 1: 3D view
        ax1 = self.fig.add_subplot(221, projection='3d')
        self.plot_3d(ax1)
        
        # Figure 2: Maximum XY cross-section
        ax2 = self.fig.add_subplot(222)
        self.plot_xy_section(ax2)
        
        # Figure 3: XOZ projection
        ax3 = self.fig.add_subplot(223)
        self.plot_xoz_projection(ax3)
        
        # Figure 4: YOZ projection
        ax4 = self.fig.add_subplot(224)
        self.plot_yoz_projection(ax4)
        
        plt.tight_layout(pad=2, w_pad=3, h_pad=3)
        self.canvas.draw()
    
    def plot_3d(self, ax):
        """Plot 3D view of valid area"""
        if len(self.valid_points) > 0:
            # Use intensity values for color coding
            scatter = ax.scatter(self.valid_points[:,0], self.valid_points[:,1], self.valid_points[:,2], 
                       c=self.valid_intensities, cmap='viridis', marker='.', alpha=0.8, 
                       label=f'Valid Area (≥{self.min_required_lamps} lights)')
            cbar = plt.colorbar(scatter, ax=ax, pad=0.4)
            cbar.set_label('Intensity')
            
            if self.nearest_point is not None:
                ax.scatter(self.nearest_point[0], self.nearest_point[1], self.nearest_point[2], 
                           c='r', marker='o', s=150, edgecolors='black',
                           label=f'Nearest Point\n{self.nearest_point.round(2)}')
            if self.farthest_point is not None:
                ax.scatter(self.farthest_point[0], self.farthest_point[1], self.farthest_point[2], 
                           c='g', marker='o', s=150, edgecolors='black',
                           label=f'Farthest Point\n{self.farthest_point.round(2)}')
        
        # Mark light source positions
        lamp_positions = np.array([lamp['position'] for lamp in self.lamps])
        ax.scatter(lamp_positions[:,0], lamp_positions[:,1], lamp_positions[:,2], 
                   c='orange', marker='o', s=100, edgecolors='black', label='Light Sources')
        
        # Use user-defined coordinate ranges for visualization
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(self.z_min, self.z_max)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Valid Observation Area (Volume: {self.volume:.2f} m³)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return ax
    
    def plot_xy_section(self, ax):
        """Plot maximum XY cross-section"""
        if self.max_xy_points is not None:
            z_mask = (self.valid_points[:, 2] == self.max_xy_z)
            z_intensities = self.valid_intensities[z_mask]  # Use intensity values for color mapping
            scatter = ax.scatter(self.max_xy_points[:,0], self.max_xy_points[:,1], 
                                 c=z_intensities, cmap='viridis', marker='.', alpha=0.8)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Intensity')
        
        # Projection of light sources on XY plane
        lamp_positions = np.array([lamp['position'] for lamp in self.lamps])
        ax.scatter(lamp_positions[:,0], lamp_positions[:,1], 
                   c='orange', marker='o', s=100, edgecolors='black', label='Light Sources')
        ax.set_title(f'Max XY Section (z={self.max_xy_z:.1f}, Area={self.max_xy_area:.2f})')
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.legend()
        return ax
    
    def plot_xoz_projection(self, ax):
        """Plot XOZ projection"""
        if len(self.valid_points) > 0:
            # Use intensity values for color mapping
            scatter = ax.scatter(self.valid_points[:,0], self.valid_points[:,2], 
                                 c=self.valid_intensities, cmap='viridis', marker='.', alpha=0.8)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Intensity')
            # Projection of light sources on XOZ plane
            lamp_positions = np.array([lamp['position'] for lamp in self.lamps])
            ax.scatter(lamp_positions[:,0], lamp_positions[:,2], 
                       c='orange', marker='o', s=100, edgecolors='black', label='Light Sources')
        ax.set_title(f'XOZ Projection (Area={self.xoz_area:.2f})')
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.z_min, self.z_max)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.legend()
        return ax
    
    def plot_yoz_projection(self, ax):
        """Plot YOZ projection"""
        if len(self.valid_points) > 0:
            # Use intensity values for color mapping
            scatter = ax.scatter(self.valid_points[:,1], self.valid_points[:,2], 
                                 c=self.valid_intensities, cmap='viridis', marker='.', alpha=0.8)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Intensity')
            # Projection of light sources on YOZ plane
            lamp_positions = np.array([lamp['position'] for lamp in self.lamps])
            ax.scatter(lamp_positions[:,1], lamp_positions[:,2], 
                       c='orange', marker='o', s=100, edgecolors='black', label='Light Sources')
        ax.set_title(f'YOZ Projection (Area={self.yoz_area:.2f})')
        ax.set_xlim(self.y_min, self.y_max)
        ax.set_ylim(self.z_min, self.z_max)
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('Z (m)')
        ax.legend()
        return ax
    
    def show_individual_plot(self, plot_type):
        """Show individual plot in a new window"""
        if len(self.valid_points) == 0:
            messagebox.showinfo("Info", "No data to display, run simulation first")
            return
            
        # Create new window
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Plot {plot_type} - Underwater Lighting Simulation")
        plot_window.geometry("800x600")
        
        # Create figure
        fig = plt.figure(figsize=(7, 5), dpi=100)
        
        # Plot based on type
        if plot_type == 1:
            plot_window.title("3D View - Underwater Lighting Simulation")
            ax = fig.add_subplot(111, projection='3d')
            self.plot_3d(ax)
            default_filename = "3d_view.png"
        elif plot_type == 2:
            plot_window.title("Maximum XY Section - Underwater Lighting Simulation")
            ax = fig.add_subplot(111)
            self.plot_xy_section(ax)
            default_filename = "xy_section.png"
        elif plot_type == 3:
            plot_window.title("XOZ Projection - Underwater Lighting Simulation")
            ax = fig.add_subplot(111)
            self.plot_xoz_projection(ax)
            default_filename = "xoz_projection.png"
        elif plot_type == 4:
            plot_window.title("YOZ Projection - Underwater Lighting Simulation")
            ax = fig.add_subplot(111)
            self.plot_yoz_projection(ax)
            default_filename = "yoz_projection.png"
        else:
            plot_window.destroy()
            return
            
        plt.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
        # Add save button
        btn_frame = ttk.Frame(plot_window)
        btn_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(btn_frame, text="Save This Plot", 
                  command=lambda: self.save_figure(fig, default_filename)).pack(side=tk.RIGHT)

if __name__ == "__main__":
    root = tk.Tk()
    app = UnderwaterLightingSimulator(root)
    root.mainloop()
