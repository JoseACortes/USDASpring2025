import pandas as pd
import numpy as np

from analysispackage import plotter as pt
from analysispackage import classicanalysis as cla

# Import
import pandas as pd
import numpy as np
import INS_Analysis as insd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import linregress

# In[1]: Loading Data

generator_path = '../0DataGeneration/gen/' 
trials = generator_path+'trials.csv'
trials = pd.read_csv(trials, index_col=0)

true_c_concentrations = trials['6000'].values
true_c_concentrations = np.array([float(x) for x in true_c_concentrations])
true_c_concentrations = -true_c_concentrations
true_c_concentrations = true_c_concentrations.tolist()

true_si_concentrations = trials['14000'].values
true_si_concentrations = np.array([float(x) for x in true_si_concentrations])
true_si_concentrations = -true_si_concentrations
true_si_concentrations = true_si_concentrations.tolist()

concentrations = [[si, c] for c, si in zip(true_c_concentrations, true_si_concentrations)]

spectrums = generator_path+'spectrums.npz'
spectrums = np.load(spectrums)
bins = spectrums['x']
spectrums = spectrums['y'][:, 0, :]
headers = [str(x[1])[2:][:2]+'.'+str(x[1])[2:][2:4]+"%C" for x in concentrations]
df = pd.DataFrame(spectrums.T, columns=headers)


# In[2]: Plotting
# pt.general_plotter(
#     df, 
#     bins=bins, 
#     suptitle='Carbon and Silicon Concentrations', 
#     c_window=[4.2, 4.7], 
#     si_window=[1.6, 1.95], 
#     low_window=[0, 0.5], 
#     output_folder='output/'
#     )

# pt.single_spectrum(
#     df, 
#     bins=bins,
#     column=8, 
#     suptitle='Example Spectrum', 
#     c_window=[4.2, 4.7], 
#     si_window=[1.6, 1.95],
#     output_folder='output/'
#     )


# %% Peak Fitting
fitting_df, c_lines_df, si_lines_df = cla.PerpendicularDrop(
    df,
    bins=bins,
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    )

# train_index = [0, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# test_index = [1, 2, 4, 5]
train_index = [0, 15]
test_index = [1, 2, 4, 5,3, 6, 7, 8, 9, 10, 11, 12, 13, 14]

train_mask = np.array([True if i in train_index else False for i in range(len(fitting_df))])
test_mask = np.array([True if i in test_index else False for i in range(len(fitting_df))])

training_x = fitting_df['Carbon Peak Area'].iloc[train_index]
training_y = np.array(true_c_concentrations)[train_index]
testing_x = fitting_df['Carbon Peak Area'].iloc[test_index]
testing_y = np.array(true_c_concentrations)[test_index]

result = linregress(training_x, training_y)

analyze = lambda x: result.slope * x + result.intercept
x_hat = fitting_df['Carbon Peak Area'].apply(analyze)
x_hat = np.array(x_hat)
mae_test = np.mean(np.abs(testing_y - analyze(testing_x)))



# %% Plotting Peak Fitting Results

# pt.plot_reg_results(
#     x_hat,
#     true_c_concentrations,
#     element='Carbon',
#     training_mask=train_mask,
#     suptitle='Prediction Results', 
#     output_folder='output/',
#     filetype='png'
#     )

# pt.plot_fitting_results(
#     fitting_df,
#     true_c_concentrations,
#     column=8,
#     suptitle='Fitting Results', 
#     output_folder='output/',
#     filetype='png'
#     )

pt.single_carbon_fit_plotter(
    c_lines_df[c_lines_df.columns[1:]], 
    column=8,
    bins=c_lines_df['bins'], 
    suptitle='Perpendicular Drop Example', 
    c_window=[4.2, 4.7], 
    output_folder='output/',
    filetype='png'
    )



# %% Tangent Skim Fitting
fitting_df, c_lines_df, si_lines_df = cla.TangentSkim(
    df,
    bins=bins,
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    )

# train_index = [0, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# test_index = [1, 2, 4, 5]
train_index = [0, 15]
test_index = [1, 2, 4, 5,3, 6, 7, 8, 9, 10, 11, 12, 13, 14]

train_mask = np.array([True if i in train_index else False for i in range(len(fitting_df))])
test_mask = np.array([True if i in test_index else False for i in range(len(fitting_df))])

training_x = fitting_df['Carbon Peak Area'].iloc[train_index]
training_y = np.array(true_c_concentrations)[train_index]
testing_x = fitting_df['Carbon Peak Area'].iloc[test_index]
testing_y = np.array(true_c_concentrations)[test_index]

result = linregress(training_x, training_y)

analyze = lambda x: result.slope * x + result.intercept
x_hat = fitting_df['Carbon Peak Area'].apply(analyze)
x_hat = np.array(x_hat)
mae_test = np.mean(np.abs(testing_y - analyze(testing_x)))



# %% Plotting Peak Fitting Results

# pt.plot_reg_results(
#     x_hat,
#     true_c_concentrations,
#     element='Carbon',
#     training_mask=train_mask,
#     suptitle='Prediction Results', 
#     output_folder='output/',
#     filetype='png'
#     )

# pt.plot_fitting_results(
#     fitting_df,
#     true_c_concentrations,
#     column=8,
#     suptitle='Fitting Results', 
#     output_folder='output/',
#     filetype='png'
#     )

pt.single_carbon_fit_plotter(
    c_lines_df[c_lines_df.columns[1:]], 
    column=8,
    bins=c_lines_df['bins'], 
    suptitle='Tangent Skim Example', 
    c_window=[4.2, 4.7], 
    output_folder='output/',
    filetype='png'
    )