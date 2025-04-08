import pandas as pd
import numpy as np

from analysispackage import plotter as pt
from analysispackage import peakfitanalysis as pfa
from analysispackage import componentanalysis as ca
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

true_c_concentrations = np.array(true_c_concentrations)


# train_index = [0, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# test_index = [1, 2, 4, 5]
train_index = [0, 15]
test_index = [1, 2, 4, 5,3, 6, 7, 8, 9, 10, 11, 12, 13, 14]
train_mask = np.array([True if i in train_index else False for i in range(len(df.columns))])
test_mask = np.array([True if i in test_index else False for i in range(len(df.columns))])

fitting_df, predicted_df = ca.Analyze(
    df,
    train_mask=train_mask,
    true_c_concentrations=true_c_concentrations,
    normalize=True,
)

training_x = predicted_df['Carbon Portion'].iloc[train_index]
training_y = np.array(true_c_concentrations)[train_index]
testing_x = predicted_df['Carbon Portion'].iloc[test_index]
testing_y = np.array(true_c_concentrations)[test_index]

# %% Fitting Components
pt.single_component_fit(
    df,
    df[df.columns[train_mask]],
    fitting_df,
    bins,
    column=8,
    suptitle='Component Fitting',
    output_folder='output/',
    window=[4.2, 4.7],
    filetype='png'
)

result = linregress(training_x, training_y)

analyze = lambda x: result.slope * x + result.intercept
x_hat = predicted_df['Carbon Portion'].apply(analyze)
x_hat = np.array(x_hat)
mae_test = np.mean(np.abs(testing_y - analyze(testing_x)))

# %% Plotting Peak Fitting Results

# pt.plot_reg_results(
#     x_hat,
#     true_c_concentrations,
#     element='Carbon',
#     training_mask=train_mask,
#     suptitle='Component Fitting Results', 
#     output_folder='output/'
#     )