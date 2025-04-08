import pandas as pd
import numpy as np
from tabulate import tabulate

from analysispackage import plotter as pt
from analysispackage import peakfitanalysis as pfa
from analysispackage import componentanalysis as ca
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


# pt.general_plotter(
#     df, 
#     bins=bins, 
#     suptitle='Carbon and Silicon Concentrations', 
#     c_window=[4.2, 4.7], 
#     si_window=[1.6, 1.95], 
#     low_window=[0, 0.5], 
#     output_folder='output/'
#     )

test_names = []
mses = []
r2s = []
predicted_concentrations = []

train_index = [0, 15]
test_index = [1, 2, 4, 5,3, 6, 7, 8, 9, 10, 11, 12, 13, 14]


# %% single Peak Fitting
fitting_df, carbon_fitting_df, si_fitting_df, c_lines_df, si_lines_df = pfa.PeakFit(
    df,
    bins=bins,
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    c_baseline='linear',
    si_baseline='linear',
    )

train_mask = np.array([True if i in train_index else False for i in range(len(fitting_df))])
test_mask = np.array([True if i in test_index else False for i in range(len(fitting_df))])

training_x = fitting_df['Carbon Peak Area'].iloc[train_index]
training_y = np.array(true_c_concentrations)[train_index]
testing_x = fitting_df['Carbon Peak Area'].iloc[test_index]
testing_y = np.array(true_c_concentrations)[test_index]

result = linregress(training_x, training_y)

# pt.fits_plotter(
#     df,
#     c_lines_df[c_lines_df.columns[1:]],
#     si_lines_df[si_lines_df.columns[1:]],
#     c_bins=c_lines_df['bins'],
#     si_bins=si_lines_df['bins'],
#     suptitle='Single Peak - Carbon and Silicone Fits',
#     output_folder='output/'
#     )

analyze = lambda x: result.slope * x + result.intercept
x_hat = fitting_df['Carbon Peak Area'].apply(analyze)
x_hat = np.array(x_hat)
mse_test = np.mean(np.square(testing_y - analyze(testing_x)))
r2 = result.rvalue**2
test_names.append('PF - Linear Baseline')
mses.append(mse_test)
r2s.append(r2)
predicted_concentrations.append(x_hat)

# %% exp falloff single Peak Fitting
fitting_df, carbon_fitting_df, si_fitting_df, c_lines_df, si_lines_df = pfa.PeakFit(
    df,
    bins=bins,
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    c_baseline='exp_falloff',
    si_baseline='exp_falloff',
    )

train_mask = np.array([True if i in train_index else False for i in range(len(fitting_df))])
test_mask = np.array([True if i in test_index else False for i in range(len(fitting_df))])

training_x = fitting_df['Carbon Peak Area'].iloc[train_index]
training_y = np.array(true_c_concentrations)[train_index]
testing_x = fitting_df['Carbon Peak Area'].iloc[test_index]
testing_y = np.array(true_c_concentrations)[test_index]

result = linregress(training_x, training_y)

# pt.fits_plotter(
#     df,
#     c_lines_df[c_lines_df.columns[1:]],
#     si_lines_df[si_lines_df.columns[1:]],
#     c_bins=c_lines_df['bins'],
#     si_bins=si_lines_df['bins'],
#     suptitle='Exponential Falloff - Carbon and Silicone Fits',
#     output_folder='output/'
#     )

analyze = lambda x: result.slope * x + result.intercept
x_hat = fitting_df['Carbon Peak Area'].apply(analyze)
x_hat = np.array(x_hat)
mse_test = np.mean(np.square(testing_y - analyze(testing_x)))
r2 = result.rvalue**2
test_names.append('PF - Exp Falloff Baseline')
mses.append(mse_test)
r2s.append(r2)
print(result.rvalue)
predicted_concentrations.append(x_hat)


# %% component analysis


true_c_concentrations = np.array(true_c_concentrations)


fitting_df, predicted_df = ca.Analyze(
    df,
    train_mask=train_mask,
    true_c_concentrations=true_c_concentrations,
    normalize=False,
)

training_x = predicted_df['Carbon Portion'].iloc[train_index]
training_y = np.array(true_c_concentrations)[train_index]
testing_x = predicted_df['Carbon Portion'].iloc[test_index]
testing_y = np.array(true_c_concentrations)[test_index]

result = linregress(training_x, training_y)

analyze = lambda x: result.slope * x + result.intercept
x_hat = predicted_df['Carbon Portion'].apply(analyze)
x_hat = np.array(x_hat)
mae_test = np.mean(np.square(testing_y - analyze(testing_x)))

test_names.append('CF')
mses.append(mae_test)
r2s.append(result.rvalue**2)
predicted_concentrations.append(x_hat)


# %% Perpendicular Drop Fitting
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
test_names.append('PD')
mses.append(mae_test)
r2s.append(result.rvalue**2)
predicted_concentrations.append(x_hat)

# tangent skim
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
test_names.append('TS')
mses.append(mae_test)
r2s.append(result.rvalue**2)
predicted_concentrations.append(x_hat)

results = pd.DataFrame({'Test': test_names, 'MSE': mses, 'R2': r2s})
# print table
print(tabulate(results, headers='keys', tablefmt='plain'))


# %% Plotting Peak Fitting Results

plt.figure(frameon=False, figsize=(6.3, 6.3))
plt.plot(true_c_concentrations, true_c_concentrations, label='Ideal', linestyle='dashed', color='black', linewidth=2, alpha=0.1)
for i in range(len(test_names)):
    plt.scatter(true_c_concentrations, predicted_concentrations[i], label=results.Test[i], marker='x', s = 150)

plt.xlabel('True Concentration')
plt.ylabel('Predicted Concentration')
plt.title('True vs Predicted Concentrations')
plt.legend()
# sqare the plot
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.savefig('output/true_vs_predicted.png')