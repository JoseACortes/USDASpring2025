try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy

except ImportError:
    # give error message if the package is not found
    print("One or more required packages (numpy, pandas, matplotlib) are not found. Please make sure they are installed and available in your Python path.")
    raise


try:
    import CortesAnalysisPackage as cap
    print("CortesAnalysisPackage imported successfully.")
except ImportError:
    # give error message if the package is not found
    print("CortesAnalysisPackage not found. Please make sure it is installed and available in your Python path or in this folder.")
    raise

try:
    import CortesAnalysisPackage.readfiles as rf
    print("readfiles module imported successfully.")
    # readfiles.readexcel
except ImportError:
    # give error message if the package is not found
    print("readfiles module not found. Please make sure it is installed and available in your Python path or in this folder.")
    raise

# Test Data Import
try:
    # Assuming the Excel file is in the same directory as this script
    file_path = "data/test/test_excel.xlsx"  # Replace with your actual file path
    df = rf.readexcel(file_path)
    print("Excel file read successfully.")
    print(df.head())
except FileNotFoundError:
    # give error message if the file is not found
    print(f"File {file_path} not found. Please make sure it exists in the specified path.")
    raise
except Exception as e:
    # give error message if there is any other error
    print(f"An error occurred while reading the Excel file: {e}")
    raise

# Test Data Import
try:
    # Assuming the CSV file is in the same directory as this script
    file_path = "data/test/test_csv.csv"  # Replace with your actual file path
    df = rf.readcsv(file_path)
    print("CSV file read successfully.")
    print(df.head())
except FileNotFoundError:
    # give error message if the file is not found
    print(f"File {file_path} not found. Please make sure it exists in the specified path.")
    raise
except Exception as e:
    # give error message if there is any other error
    print(f"An error occurred while reading the CSV file: {e}")
    raise

try:
    bins = np.arange(len(df))
    # or
    bins = pd.read_csv("data/test/test_bins.csv")['bins'].values.flatten()
    print("Bins read successfully.")
    print("Bins shape:", bins.shape)
except FileNotFoundError:
    # give error message if the file is not found
    print(f"File {file_path} not found. Please make sure it exists in the specified path.")
    raise
except Exception as e:
    # give error message if there is any other error
    print(f"An error occurred while reading the bins file: {e}")
    raise

try:
    import CortesAnalysisPackage.peakfitting as pfa
    import CortesAnalysisPackage.componentfitting as cfa
    import CortesAnalysisPackage.classical as cla
    import CortesAnalysisPackage.plot as pt
    print("modules imported successfully.")
except ImportError:
    # give error message if the package is not found
    print("modules not found. Please make sure it is installed and available in your Python path or in this folder.")
    raise

c_window = [4.2, 4.7]
si_window = [1.6, 1.95]
c_bins = bins[(bins >= c_window[0]) & (bins <= c_window[1])]
si_bins = bins[(bins >= si_window[0]) & (bins <= si_window[1])]

fitting_df, carbon_fitting_df, si_fitting_df, c_lines_df, si_lines_df = pfa.PeakFit(
    df,
    bins=bins,
    c_window=c_window,
    si_window=si_window,
    c_baseline='linear',
    si_baseline='linear',
    )

pt.fits_plotter(
    df,
    c_lines_df,
    si_lines_df,
    c_bins,
    si_bins,
    suptitle='Carbon and Silicon Concentrations',
    filename='output/test/test_PeakFit_plotter.png',)

with pd.ExcelWriter("output/test/test_PeakFit.xlsx", engine='openpyxl') as writer:
    fitting_df.to_excel(writer, index=False, sheet_name='Peak Fitting')
    carbon_fitting_df.to_excel(writer, index=False, sheet_name='Carbon Fitting')
    si_fitting_df.to_excel(writer, index=False, sheet_name='Silicone Fitting')
    c_lines_df.to_excel(writer, index=False, sheet_name='C Lines')
    si_lines_df.to_excel(writer, index=False, sheet_name='Si Lines')


fitting_df = cfa.Decompose(
        df,
        train_cols=df.columns[1:4],
        normalize=True,
        convex_regression=True,
    )

fitting_df.to_excel("output/test/test_Decompose.xlsx", index=False, sheet_name='Decompose')

fitting_df, c_lines_df, si_lines_df = cla.PerpendicularDrop(
    df,
    bins=bins,
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    )

pt.fits_plotter(
    df,
    c_lines_df,
    si_lines_df,
    c_bins,
    si_bins,
    suptitle='Carbon and Silicon Concentrations',
    filename='output/test/test_PerpendicularDrop_plotter.png',)


with pd.ExcelWriter("output/test/test_PerpendicularDrop.xlsx", engine='openpyxl') as writer:
    fitting_df.to_excel(writer, index=False, sheet_name='Perpendicular Drop')
    c_lines_df.to_excel(writer, index=False, sheet_name='C Lines')
    si_lines_df.to_excel(writer, index=False, sheet_name='Si Lines')


fitting_df, c_lines_df, si_lines_df = cla.TangentSkim(
    df,
    bins=bins,
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    )

pt.fits_plotter(
    df,
    c_lines_df,
    si_lines_df,
    c_bins,
    si_bins,
    suptitle='Carbon and Silicon Concentrations',
    filename='output/test/test_TangentSkim_plotter.png',)

with pd.ExcelWriter("output/test/test_TangentSkim.xlsx", engine='openpyxl') as writer:
    fitting_df.to_excel(writer, index=False, sheet_name='Tangent Skim')
    c_lines_df.to_excel(writer, index=False, sheet_name='C Lines')
    si_lines_df.to_excel(writer, index=False, sheet_name='Si Lines')

pt.multi_spectrum(
    df,
    bins=bins,
    suptitle='Carbon and Silicon Concentrations', 
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95], 
    filename='output/test/test_multi_spectrum.png'
)
