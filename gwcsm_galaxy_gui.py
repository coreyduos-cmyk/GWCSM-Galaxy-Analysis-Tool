import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import requests
from io import StringIO
from threading import Thread
import queue

# --- GLOBAL CONFIGURATION ---
BASE_PATH = os.getcwd()
DATA_DIR = os.path.join(BASE_PATH, 'galaxy_data')
RESULTS_DIR = os.path.join(BASE_PATH, 'results')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# SDSS DR18 SQL Query URL
SDSS_SQL_URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"

# Constants
H0 = 73  # Hubble constant in km/s/Mpc
C = 299792.458  # Speed of light in km/s

# =================================================================
# UTILITY FUNCTIONS
# =================================================================

def redshift_to_distance(z):
    """Convert redshift to comoving distance in Mpc"""
    return (C * z) / H0

def calculate_cartesian(ra, dec, z):
    """Convert RA, Dec, redshift to 3D Cartesian coordinates"""
    d = redshift_to_distance(z)
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
   
    x = d * np.cos(dec_rad) * np.cos(ra_rad)
    y = d * np.cos(dec_rad) * np.sin(ra_rad)
    z_coord = d * np.sin(dec_rad)
   
    return x, y, z_coord

def query_sdss(ra_min, ra_max, dec_min, dec_max, z_min, z_max, max_rows=50000):
    """Query SDSS for galaxy data within specified ranges"""
    sql_query = f"""
    SELECT TOP {max_rows}
        objID, ra, dec, z
    FROM SpecPhoto
    WHERE
        ra BETWEEN {ra_min} AND {ra_max}
        AND dec BETWEEN {dec_min} AND {dec_max}
        AND z BETWEEN {z_min} AND {z_max}
        AND class = 'GALAXY'
        AND zWarning = 0
    ORDER BY ra
    """
   
    try:
        response = requests.get(SDSS_SQL_URL, params={'cmd': sql_query, 'format': 'csv'}, timeout=60)
        response.raise_for_status()
       
        # Parse CSV response - skip comment lines
        lines = response.text.strip().split('\n')
        csv_lines = [line for line in lines if not line.startswith('#')]
        csv_text = '\n'.join(csv_lines)
        
        if len(csv_lines) <= 1:
            raise Exception("Query returned no data - try different coordinates or redshift range")
            
        data = pd.read_csv(StringIO(csv.text))
        return data
    except Exception as e:
        raise Exception(f"SDSS query failed: {str(e)}")

# =================================================================
# GUI APPLICATION
# =================================================================

class GWCSMGalaxyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GWCSM Galaxy Analysis Tool")
        self.root.geometry("1100x800")
       
        # Data storage
        self.current_data = None
        self.output_queue = queue.Queue()
       
        # Create UI
        self.create_widgets()
       
        # Start queue monitor
        self.root.after(100, self.monitor_output_queue)
   
    def create_widgets(self):
        """Create all GUI widgets"""
        # Title
        title_label = tk.Label(self.root, text="🌌 GWCSM GALAXY ANALYSIS TOOL",
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
       
        info_label = tk.Label(self.root,
                             text="Analyze gravitational wave patterns in galaxy distributions",
                             font=("Arial", 9))
        info_label.pack()
       
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
       
        # Tab 1: Load Existing CSV
        self.load_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.load_tab, text="Load Existing CSV")
        self.create_load_tab()
       
        # Tab 2: Query SDSS
        #self.query_tab = ttk.Frame(self.notebook)
        #self.notebook.add(self.query_tab, text="Query SDSS")
        #self.create_query_tab()
       
        # Tab 3: Analysis
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Analysis & Visualization")
        self.create_analysis_tab()
       
        # Output Log (bottom of window)
        output_frame = ttk.LabelFrame(self.root, text="Output Log", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
       
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10,
                                                     font=("Consolas", 9))
        self.output_text.pack(fill=tk.BOTH, expand=True)
       
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
   
    def create_load_tab(self):
        """Create the Load CSV tab"""
        frame = ttk.Frame(self.load_tab, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
       
        # File selection
        ttk.Label(frame, text="Select CSV File:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w", pady=10)
       
        file_frame = ttk.Frame(frame)
        file_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)
       
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Load", command=self.load_csv).pack(side=tk.LEFT, padx=5)
       
        # Data filtering options
        ttk.Label(frame, text="Data Filtering:", font=("Arial", 11, "bold")).grid(row=2, column=0, sticky="w", pady=(20, 10))
       
        # Sky coverage mode
        self.sky_mode_var = tk.StringVar(value="whole")
        ttk.Radiobutton(frame, text="Whole Sky (use all data)", variable=self.sky_mode_var,
                       value="whole", command=self.update_filter_mode).grid(row=3, column=0, sticky="w", pady=5)
        ttk.Radiobutton(frame, text="Specific Slice", variable=self.sky_mode_var,
                       value="slice", command=self.update_filter_mode).grid(row=4, column=0, sticky="w", pady=5)
       
        # Slice parameters (initially disabled)
        self.slice_frame = ttk.LabelFrame(frame, text="Slice Parameters", padding="10")
        self.slice_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)
       
        # RA range
        ttk.Label(self.slice_frame, text="RA Range (degrees):").grid(row=0, column=0, sticky="w", pady=5)
        self.ra_min_var = tk.DoubleVar(value=0)
        self.ra_max_var = tk.DoubleVar(value=360)
        ttk.Entry(self.slice_frame, textvariable=self.ra_min_var, width=15).grid(row=0, column=1, padx=5)
        ttk.Label(self.slice_frame, text="to").grid(row=0, column=2)
        ttk.Entry(self.slice_frame, textvariable=self.ra_max_var, width=15).grid(row=0, column=3, padx=5)
       
        # Dec range
        ttk.Label(self.slice_frame, text="Dec Range (degrees):").grid(row=1, column=0, sticky="w", pady=5)
        self.dec_min_var = tk.DoubleVar(value=-90)
        self.dec_max_var = tk.DoubleVar(value=90)
        ttk.Entry(self.slice_frame, textvariable=self.dec_min_var, width=15).grid(row=1, column=1, padx=5)
        ttk.Label(self.slice_frame, text="to").grid(row=1, column=2)
        ttk.Entry(self.slice_frame, textvariable=self.dec_max_var, width=15).grid(row=1, column=3, padx=5)
       
        # Redshift range
        ttk.Label(self.slice_frame, text="Redshift Range:").grid(row=2, column=0, sticky="w", pady=5)
        self.z_min_load_var = tk.DoubleVar(value=0.0)
        self.z_max_load_var = tk.DoubleVar(value=1.0)
        ttk.Entry(self.slice_frame, textvariable=self.z_min_load_var, width=15).grid(row=2, column=1, padx=5)
        ttk.Label(self.slice_frame, text="to").grid(row=2, column=2)
        ttk.Entry(self.slice_frame, textvariable=self.z_max_load_var, width=15).grid(row=2, column=3, padx=5)
       
        # Initially disable slice parameters
        self.update_filter_mode()
       
        # Apply filter button
        ttk.Button(frame, text="Apply Filter & Load Data", command=self.apply_filter).grid(row=6, column=0, pady=20)
   
    def create_query_tab(self):
        """Create the Query SDSS tab"""
        frame = ttk.Frame(self.query_tab, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
       
        ttk.Label(frame, text="SDSS DR18 Query Parameters", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
       
        # RA range
        ttk.Label(frame, text="RA Range (degrees):").grid(row=1, column=0, sticky="w", pady=5)
        self.query_ra_min_var = tk.DoubleVar(value=240)
        self.query_ra_max_var = tk.DoubleVar(value=255)
        ttk.Entry(frame, textvariable=self.query_ra_min_var, width=15).grid(row=1, column=1, padx=5)
        ttk.Label(frame, text="to").grid(row=1, column=2)
        ttk.Entry(frame, textvariable=self.query_ra_max_var, width=15).grid(row=1, column=3, padx=5)
       
        # Dec range
        ttk.Label(frame, text="Dec Range (degrees):").grid(row=2, column=0, sticky="w", pady=5)
        self.query_dec_min_var = tk.DoubleVar(value=40)
        self.query_dec_max_var = tk.DoubleVar(value=45)
        ttk.Entry(frame, textvariable=self.query_dec_min_var, width=15).grid(row=2, column=1, padx=5)
        ttk.Label(frame, text="to").grid(row=2, column=2)
        ttk.Entry(frame, textvariable=self.query_dec_max_var, width=15).grid(row=2, column=3, padx=5)
       
        # Redshift range
        ttk.Label(frame, text="Redshift Range:").grid(row=3, column=0, sticky="w", pady=5)
        self.query_z_min_var = tk.DoubleVar(value=0.15)
        self.query_z_max_var = tk.DoubleVar(value=0.65)
        ttk.Entry(frame, textvariable=self.query_z_min_var, width=15).grid(row=3, column=1, padx=5)
        ttk.Label(frame, text="to").grid(row=3, column=2)
        ttk.Entry(frame, textvariable=self.query_z_max_var, width=15).grid(row=3, column=3, padx=5)
       
        # Max rows per query
        ttk.Label(frame, text="Max Rows per Query:").grid(row=4, column=0, sticky="w", pady=5)
        self.max_rows_var = tk.IntVar(value=50000)
        ttk.Entry(frame, textvariable=self.max_rows_var, width=15).grid(row=4, column=1, padx=5)
        ttk.Label(frame, text="(safety limit)", font=("Arial", 8)).grid(row=4, column=2, columnspan=2, sticky="w")
       
        # Query button
        ttk.Button(frame, text="🔍 Query SDSS", command=self.query_sdss_data).grid(row=5, column=0, columnspan=2, pady=20)
       
        # Save location
        ttk.Label(frame, text="Save As:").grid(row=6, column=0, sticky="w", pady=5)
        self.save_filename_var = tk.StringVar(value="sdss_query_result.csv")
        ttk.Entry(frame, textvariable=self.save_filename_var, width=40).grid(row=6, column=1, columnspan=3, sticky="ew", padx=5)
   
    def create_analysis_tab(self):
        """Create the Analysis tab"""
        frame = ttk.Frame(self.analysis_tab, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
       
        ttk.Label(frame, text="Analysis Options", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
       
        # Data info
        self.data_info_var = tk.StringVar(value="No data loaded")
        ttk.Label(frame, textvariable=self.data_info_var, font=("Arial", 10)).grid(row=1, column=0, columnspan=2, pady=10)
       
        # Histogram bins
        ttk.Label(frame, text="Histogram Bins:").grid(row=2, column=0, sticky="w", pady=5)
        self.bins_var = tk.IntVar(value=50)
        ttk.Spinbox(frame, from_=10, to=200, textvariable=self.bins_var, width=15).grid(row=2, column=1, sticky="w", padx=5)
       
        # Visualization buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
       
        ttk.Button(button_frame, text="📊 Redshift Histogram", command=self.plot_histogram).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🌐 3D Scatter Plot", command=self.plot_3d_scatter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📈 Wave Analysis", command=self.analyze_wave_pattern).pack(side=tk.LEFT, padx=5)
       
        # Export options
        export_frame = ttk.LabelFrame(frame, text="Export", padding="10")
        export_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=20)
       
        ttk.Button(export_frame, text="💾 Export Filtered Data", command=self.export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="📁 Open Results Folder", command=self.open_results_folder).pack(side=tk.LEFT, padx=5)
   
    def update_filter_mode(self):
        """Enable/disable slice parameters based on mode"""
        if self.sky_mode_var.get() == "whole":
            # Disable all slice inputs
            for child in self.slice_frame.winfo_children():
                if isinstance(child, ttk.Entry):
                    child.config(state='disabled')
        else:
            # Enable all slice inputs
            for child in self.slice_frame.winfo_children():
                if isinstance(child, ttk.Entry):
                    child.config(state='normal')
   
    def browse_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(
            initialdir=DATA_DIR,
            title="Select Galaxy Data CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.file_path_var.set(filename)
   
    def load_csv(self):
        """Load CSV file"""
        filepath = self.file_path_var.get()
        if not filepath:
            messagebox.showwarning("No File", "Please select a CSV file first")
            return
       
        try:
            self.log(f"Loading {os.path.basename(filepath)}...")
            data = pd.read_csv(filepath)
            # Normalize column names to lowercase
            data.columns = data.columns.str.lower()
           
            # Verify required columns
            required_cols = ['ra', 'dec', 'z']
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
           
            self.current_data = data
            self.log(f"✅ Loaded {len(data)} galaxies")
            self.data_info_var.set(f"Loaded: {len(data)} galaxies")
            self.status_var.set(f"Loaded {len(data)} galaxies from {os.path.basename(filepath)}")
           
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
            self.log(f"❌ Error: {str(e)}")
   
    def apply_filter(self):
        """Apply filtering to loaded data"""
        if self.current_data is None:
            messagebox.showwarning("No Data", "Please load a CSV file first")
            return
       
        try:
            data = self.current_data.copy()
           
            if self.sky_mode_var.get() == "slice":
                # Apply filters
                ra_min = self.ra_min_var.get()
                ra_max = self.ra_max_var.get()
                dec_min = self.dec_min_var.get()
                dec_max = self.dec_max_var.get()
                z_min = self.z_min_load_var.get()
                z_max = self.z_max_load_var.get()
               
                data = data[
                    (data['ra'] >= ra_min) & (data['ra'] <= ra_max) &
                    (data['dec'] >= dec_min) & (data['dec'] <= dec_max) &
                    (data['z'] >= z_min) & (data['z'] <= z_max)
                ]
               
                self.log(f"Filtered to RA [{ra_min}, {ra_max}], Dec [{dec_min}, {dec_max}], z [{z_min}, {z_max}]")
            else:
                self.log("Using whole sky data (no filtering)")
           
            self.current_data = data
            self.data_info_var.set(f"Active: {len(data)} galaxies")
            self.log(f"✅ {len(data)} galaxies ready for analysis")
           
            # Switch to analysis tab
            self.notebook.select(self.analysis_tab)
           
        except Exception as e:
            messagebox.showerror("Error", f"Filter failed: {str(e)}")
            self.log(f"❌ Error: {str(e)}")
   
    def query_sdss_data(self):
        """Query SDSS in background thread"""
        ra_min = self.query_ra_min_var.get()
        ra_max = self.query_ra_max_var.get()
        dec_min = self.query_dec_min_var.get()
        dec_max = self.query_dec_max_var.get()
        z_min = self.query_z_min_var.get()
        z_max = self.query_z_max_var.get()
        max_rows = self.max_rows_var.get()
       
        self.log(f"Querying SDSS: RA [{ra_min}, {ra_max}], Dec [{dec_min}, {dec_max}], z [{z_min}, {z_max}]")
        self.status_var.set("Querying SDSS...")
       
        # Run query in background
        query_thread = Thread(target=self._execute_query,
                             args=(ra_min, ra_max, dec_min, dec_max, z_min, z_max, max_rows))
        query_thread.daemon = True
        query_thread.start()
   
    def _execute_query(self, ra_min, ra_max, dec_min, dec_max, z_min, z_max, max_rows):
        """Execute SDSS query in background"""
        try:
            data = query_sdss(ra_min, ra_max, dec_min, dec_max, z_min, z_max, max_rows)
           
            if len(data) == 0:
                self.output_queue.put("⚠️ Query returned no results")
                self.root.after(0, lambda: self.status_var.set("No results found"))
                return
           
            # Save to file
            filename = self.save_filename_var.get()
            filepath = os.path.join(DATA_DIR, filename)
            data.to_csv(filepath, index=False)
           
            self.output_queue.put(f"✅ Downloaded {len(data)} galaxies")
            self.output_queue.put(f"💾 Saved to {filename}")
           
            # Load into current data
            self.current_data = data
            self.root.after(0, lambda: self.data_info_var.set(f"Loaded: {len(data)} galaxies"))
            self.root.after(0, lambda: self.status_var.set(f"Query complete: {len(data)} galaxies"))
            self.root.after(0, lambda: self.notebook.select(self.analysis_tab))
           
        except Exception as e:
            self.output_queue.put(f"❌ Query failed: {str(e)}")
            self.root.after(0, lambda: self.status_var.set("Query failed"))
   
    def plot_histogram(self):
        """Plot redshift histogram"""
        if self.current_data is None or len(self.current_data) == 0:
            messagebox.showwarning("No Data", "No data available for plotting")
            return
       
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
           
            bins = self.bins_var.get()
            ax.hist(self.current_data['z'], bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Redshift (z)', fontsize=12)
            ax.set_ylabel('Number of Galaxies', fontsize=12)
            ax.set_title(f'Redshift Distribution ({len(self.current_data)} galaxies)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
           
            plt.tight_layout()
            plt.show()
           
            self.log(f"Generated redshift histogram with {bins} bins")
           
        except Exception as e:
            messagebox.showerror("Error", f"Plotting failed: {str(e)}")
            self.log(f"❌ Error: {str(e)}")
   
    def plot_3d_scatter(self):
        """Plot interactive 3D scatter plot"""
        if self.current_data is None or len(self.current_data) == 0:
            messagebox.showwarning("No Data", "No data available for plotting")
            return
       
        try:
            self.log("Generating 3D scatter plot (this may take a moment)...")
           
            # Calculate Cartesian coordinates
            data = self.current_data.copy()
            coords = data.apply(lambda row: calculate_cartesian(row['ra'], row['dec'], row['z']), axis=1)
            data['x'] = coords.apply(lambda c: c[0])
            data['y'] = coords.apply(lambda c: c[1])
            data['z_coord'] = coords.apply(lambda c: c[2])
           
            # Create 3D plot
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
           
            scatter = ax.scatter(data['x'], data['y'], data['z_coord'],
                               c=data['z'], cmap='viridis', marker='.', s=1, alpha=0.6)
           
            ax.set_xlabel('X (Mpc)', fontsize=10)
            ax.set_ylabel('Y (Mpc)', fontsize=10)
            ax.set_zlabel('Z (Mpc)', fontsize=10)
            ax.set_title(f'3D Galaxy Distribution ({len(data)} galaxies)\nColor = Redshift',
                        fontsize=12, fontweight='bold')
           
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label('Redshift (z)', fontsize=10)
           
            plt.tight_layout()
            plt.show()
           
            self.log(f"✅ 3D plot generated - use mouse to rotate!")
           
        except Exception as e:
            messagebox.showerror("Error", f"3D plotting failed: {str(e)}")
            self.log(f"❌ Error: {str(e)}")
   
    def analyze_wave_pattern(self):
        """Analyze wave pattern in redshift distribution"""
        if self.current_data is None or len(self.current_data) == 0:
            messagebox.showwarning("No Data", "No data available for analysis")
            return
       
        try:
            self.log("="*70)
            self.log("WAVE PATTERN ANALYSIS")
            self.log("="*70)
           
            data = self.current_data.copy()
            bins = self.bins_var.get()
           
            # Create histogram
            counts, bin_edges = np.histogram(data['z'], bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
           
            # Find peaks and troughs
            mean_count = np.mean(counts)
            std_count = np.std(counts)
           
            peaks = []
            troughs = []
           
            for i, count in enumerate(counts):
                if count > mean_count + std_count:
                    peaks.append((bin_centers[i], count))
                elif count < mean_count - std_count:
                    troughs.append((bin_centers[i], count))
           
            self.log(f"\nDataset: {len(data)} galaxies")
            self.log(f"Redshift range: z = {data['z'].min():.3f} to {data['z'].max():.3f}")
            self.log(f"Mean count per bin: {mean_count:.1f} ± {std_count:.1f}")
           
            if len(peaks) > 0:
                self.log(f"\n🔺 PEAKS (> {mean_count + std_count:.1f} galaxies):")
                for z, count in peaks:
                    sigma = (count - mean_count) / std_count
                    distance = redshift_to_distance(z)
                    self.log(f"   z = {z:.3f} ({distance:.0f} Mpc): {count:.0f} galaxies (+{sigma:.1f}σ)")
           
            if len(troughs) > 0:
                self.log(f"\n🔻 TROUGHS (< {mean_count - std_count:.1f} galaxies):")
                for z, count in troughs:
                    sigma = (mean_count - count) / std_count
                    distance = redshift_to_distance(z)
                    self.log(f"   z = {z:.3f} ({distance:.0f} Mpc): {count:.0f} galaxies (-{sigma:.1f}σ)")
           
            # Calculate wavelength if multiple peaks
            if len(peaks) >= 2:
                self.log("\n📏 WAVELENGTH ESTIMATION:")
                peaks_sorted = sorted(peaks, key=lambda x: x[0])
                for i in range(len(peaks_sorted) - 1):
                    z1, _ = peaks_sorted[i]
                    z2, _ = peaks_sorted[i + 1]
                    d1 = redshift_to_distance(z1)
                    d2 = redshift_to_distance(z2)
                    wavelength = d2 - d1
                    self.log(f"   Peak {i+1} to Peak {i+2}: λ ≈ {float(wavelength):.0f} Mpc")
           
            # Statistical significance
            if len(peaks) > 0 or len(troughs) > 0:
                self.log("\n📊 STATISTICAL SIGNIFICANCE:")
                
                peak_sigmas = [(count - mean_count) / std_count for _, count in peaks] if peaks else [0],
                trough_sigmas = [(mean_count - count) / std_count for _, count in troughs] if troughs else [0]
                max_sigma = float(max(peak_sigmas + trough_sigmas))
                
                self.log(f"   Maximum deviation: {max_sigma:.1f}σ")
                if max_sigma > 3:
                    self.log("   ✅ Highly significant structure (>3σ)")
                elif max_sigma > 2:
                    self.log("   ⚠️  Moderate significance (>2σ)")
                else:
                    self.log("   ❌ Low significance (<2σ)")
           
            self.log("="*70)
           
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.log(f"❌ Error: {str(e)}")
   
    def export_data(self):
        """Export current filtered data"""
        if self.current_data is None or len(self.current_data) == 0:
            messagebox.showwarning("No Data", "No data to export")
            return
       
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=RESULTS_DIR
            )
           
            if filename:
                self.current_data.to_csv(filename, index=False)
                self.log(f"💾 Exported {len(self.current_data)} galaxies to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Data exported to {os.path.basename(filename)}")
       
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
            self.log(f"❌ Error: {str(e)}")
   
    def open_results_folder(self):
        """Open results folder in file explorer"""
        if os.path.exists(RESULTS_DIR):
            if os.name == 'nt':  # Windows
                os.startfile(RESULTS_DIR)
            elif os.name == 'posix':  # macOS/Linux
                os.system(f'open "{RESULTS_DIR}"' if sys.platform == 'darwin' else f'xdg-open "{RESULTS_DIR}"')
        else:
            messagebox.showinfo("Info", "Results folder doesn't exist yet")
   
    def log(self, message):
        """Add message to output log"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
   
    def monitor_output_queue(self):
        """Monitor queue for messages from background threads"""
        try:
            while True:
                message = self.output_queue.get_nowait()
                self.log(message)
        except queue.Empty:
            pass
       
        self.root.after(100, self.monitor_output_queue)

# =================================================================
# MAIN ENTRY POINT
# =================================================================

if __name__ == '__main__':
    root = tk.Tk()
    app = GWCSMGalaxyGUI(root)
    root.mainloop()
