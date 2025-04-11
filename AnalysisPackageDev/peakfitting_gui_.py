import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
# Initialize Tkinter window
import CortesAnalysisPackage.peakfitting as pfa
import CortesAnalysisPackage.plot as pt
import CortesAnalysisPackage.readfiles as rf

root = tk.Tk()
root.title("CortesAnalysis GUI - Peak Fitting")

global df, folder, file, bins
df = None
folder = './'
file = None
bins = None
# import data
def select_file(): # read excel  or csv 
    global df, file
    file = filedialog.askopenfilename(filetypes=[("All files", "*.*"), ("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
    if file:
        try:
            if file.endswith('.xlsx'):
                df = rf.readexcel(file)
                return df
            elif file.endswith('.csv'):
                df = rf.readcsv(file)
                return df
            else:
                raise ValueError("Unsupported file format.")
            messagebox.showinfo("Success", "File read successfully.")
            print(df.head())
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while reading the file: {e}")
            print(f"An error occurred while reading the file: {e}")
            return None
    else:
        messagebox.showwarning("Warning", "No file selected.")
        return None

    
def select_output_folder():
    global folder
    folder = filedialog.askdirectory()
    if folder:
        try:
            # Perform operations with the selected folder
            messagebox.showinfo("Success", "Folder selected successfully.")
            print(f"Selected folder: {folder}")
            return folder
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while selecting the folder: {e}")
            print(f"An error occurred while selecting the folder: {e}")
            return None
    else:
        messagebox.showwarning("Warning", "No folder selected.")
        return None

def select_bins_file():
    global bins
    file = filedialog.askopenfilename(filetypes=[("All files", "*.*"), ("CSV files", "*.csv")])
    if file:
        try:
            bins = pd.read_csv(file)['bins'].values.flatten()
            print("Bins read successfully.")
            print("Bins shape:", bins.shape)
            return bins
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while reading the bins file: {e}")
            print(f"An error occurred while reading the bins file: {e}")
            return None
    else:
        messagebox.showwarning("Warning", "No bins file selected.")
        return None


file_path_var = tk.StringVar(value="No file selected")
bins_file_path_var = tk.StringVar(value="No bins file selected")
output_folder_path_var = tk.StringVar(value="No folder selected")

def update_file_path():
    global file  # Ensure the global variable is updated
    global df
    df = select_file()
    if df is not None:
        file_path_var.set("File selected: " + file if file else "No file selected")
        update_spinbox_ranges()  # Update spinbox ranges when df is defined

def update_bins_file_path():
    global bins
    bins = select_bins_file()
    if bins is not None:
        bins_file_path_var.set("Bins file selected: " + file if file else "No bins file selected")
        update_spinbox_ranges()  # Update spinbox ranges when bins is defined

def update_output_folder_path():
    folder = select_output_folder()
    if folder is not None:
        output_folder_path_var.set(folder)

tk.Label(root, textvariable=file_path_var, anchor="w", width=50).pack()
select_file_button = tk.Button(root, text="Select File", command=update_file_path)
select_file_button.pack()

tk.Label(root, textvariable=bins_file_path_var, anchor="w", width=50).pack()
select_bins_file_button = tk.Button(root, text="Select Bins File (Optional)", command=update_bins_file_path)
select_bins_file_button.pack()

tk.Label(root, textvariable=output_folder_path_var, anchor="w", width=50).pack()
select_output_folder_button = tk.Button(root, text="Select Output Folder", command=update_output_folder_path)
select_output_folder_button.pack()
# Peak Fitting

# Add option menus for baseline and peak functions
c_baseline_var = tk.StringVar(value='linear')
c_peak_var = tk.StringVar(value='gauss')
si_baseline_var = tk.StringVar(value='linear')
si_peak_var = tk.StringVar(value='gauss')

tk.Label(root, text="Carbon Baseline:").pack()
c_baseline_menu = tk.OptionMenu(root, c_baseline_var, 'linear', 'polynomial', 'exponential')
c_baseline_menu.pack()

tk.Label(root, text="Carbon Peak:").pack()
c_peak_menu = tk.OptionMenu(root, c_peak_var, 'gauss', 'lorentzian', 'voigt')
c_peak_menu.pack()

tk.Label(root, text="Silicon Baseline:").pack()
si_baseline_menu = tk.OptionMenu(root, si_baseline_var, 'linear', 'polynomial', 'exponential')
si_baseline_menu.pack()

tk.Label(root, text="Silicon Peak:").pack()
si_peak_menu = tk.OptionMenu(root, si_peak_var, 'gauss', 'lorentzian', 'voigt')
si_peak_menu.pack()

# Add scales for window ranges
c_window_start = tk.DoubleVar(value=0.0)
c_window_end = tk.DoubleVar(value=10.0)
si_window_start = tk.DoubleVar(value=0.0)
si_window_end = tk.DoubleVar(value=10.0)

tk.Label(root, text="Carbon Window Start:").pack()
c_window_start_spinbox = tk.Spinbox(root, from_=0, to=100, increment=0.1, textvariable=c_window_start)
c_window_start_spinbox.pack()

tk.Label(root, text="Carbon Window End:").pack()
c_window_end_spinbox = tk.Spinbox(root, from_=0, to=100, increment=0.1, textvariable=c_window_end)
c_window_end_spinbox.pack()

tk.Label(root, text="Silicon Window Start:").pack()
si_window_start_spinbox = tk.Spinbox(root, from_=0, to=100, increment=0.1, textvariable=si_window_start)
si_window_start_spinbox.pack()

tk.Label(root, text="Silicon Window End:").pack()
si_window_end_spinbox = tk.Spinbox(root, from_=0, to=100, increment=0.1, textvariable=si_window_end)
si_window_end_spinbox.pack()

def update_spinbox_ranges():
    global df, bins
    """Update the spinbox ranges and increments based on df or bins."""
    if df is not None and bins is None:
        # If df is defined, set range to 0 to number of bins - 1 with increment of 1
        num_bins = len(df.index)  # Assuming df rows represent bins
        c_window_start = tk.DoubleVar(value=0.0)
        c_window_end = tk.DoubleVar(value=num_bins - 1.0)
        c_window_start_spinbox.config(from_=0, to=num_bins - 1, increment=1, textvariable=c_window_start)
        c_window_end_spinbox.config(from_=0, to=num_bins - 1, increment=1, textvariable=c_window_end)
        si_window_start = tk.DoubleVar(value=0.0)
        si_window_end = tk.DoubleVar(value=num_bins - 1.0)
        si_window_start_spinbox.config(from_=0, to=num_bins - 1, increment=1, textvariable=si_window_start)
        si_window_end_spinbox.config(from_=0, to=num_bins - 1, increment=1, textvariable=si_window_end)
    elif bins is not None:
        # If bins is defined, set range to the values in bins with increment of 1
        min_bin = min(bins)
        max_bin = max(bins)
        num_bins = len(bins)
        c_window_start = tk.DoubleVar(value=min_bin)
        c_window_end = tk.DoubleVar(value=max_bin)
        si_window_start = tk.DoubleVar(value=min_bin)
        si_window_end = tk.DoubleVar(value=max_bin)
        c_window_start_spinbox.config(from_=min_bin, to=max_bin, increment=(max_bin - min_bin) / num_bins, textvariable=c_window_start)
        c_window_end_spinbox.config(from_=min_bin, to=max_bin, increment=(max_bin - min_bin) / num_bins, textvariable=c_window_end)
        si_window_start_spinbox.config(from_=min_bin, to=max_bin, increment=(max_bin - min_bin) / num_bins, textvariable=si_window_start)
        si_window_end_spinbox.config(from_=min_bin, to=max_bin, increment=(max_bin - min_bin) / num_bins, textvariable=si_window_end)
    else:
        # Default range if neither df nor bins is defined
        c_window_start_spinbox.config(from_=0, to=100, increment=0.1)
        c_window_end_spinbox.config(from_=0, to=100, increment=0.1)
        si_window_start_spinbox.config(from_=0, to=100, increment=0.1)
        si_window_end_spinbox.config(from_=0, to=100, increment=0.1)

# Call update_spinbox_ranges initially to set default ranges
update_spinbox_ranges()

# Function to call PeakFit with selected options
def run_peak_fit():
    c_window = (c_window_start.get(), c_window_end.get())
    si_window = (si_window_start.get(), si_window_end.get())
    c_baseline = c_baseline_var.get()
    c_peak = c_peak_var.get()
    si_baseline = si_baseline_var.get()
    si_peak = si_peak_var.get()

    if df is not None:
        try:
            result = pfa.PeakFit(
                df=df,
                c_window=c_window,
                si_window=si_window,
                bins=bins,
                c_baseline=c_baseline,
                c_peak=c_peak,
                si_baseline=si_baseline,
                si_peak=si_peak
            )
            messagebox.showinfo("Success", "Peak fitting completed successfully.")
            print("Peak fitting result:", result)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during peak fitting: {e}")
            print(f"An error occurred during peak fitting: {e}")
    else:
        messagebox.showwarning("Warning", "No data file selected.")

# Add a button to run PeakFit
run_peak_fit_button = tk.Button(root, text="Run Peak Fit", command=run_peak_fit)
run_peak_fit_button.pack()


root.mainloop()