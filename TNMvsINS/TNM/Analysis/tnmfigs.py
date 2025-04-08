import pandas as pd
import numpy as np
from analysispackage import plotter as pt
from analysispackage import peakfitanalysis as pfa
from analysispackage import componentanalysis as ca

generator_path = '../data/'
spectrums = generator_path+'spectrums.npz'
spectrums = np.load(spectrums)
bins = spectrums['x']
spectrums = spectrums['y']
detector_spectrums = spectrums

gebless_spectrums = generator_path+'gebless_spectrums.npz'
gebless_spectrums = np.load(gebless_spectrums)
gebless_spectrums = gebless_spectrums['y']

soil_fluxes = generator_path+'soil_fluxes.npz'
soil_fluxes = np.load(soil_fluxes)
soil_fluxes = soil_fluxes['y']

filenames_path = '../filenames.csv'
filenames = pd.read_csv(filenames_path)['name']

df = pd.DataFrame()
for i in range(len(filenames)):
    df[filenames[i]] = detector_spectrums[i, 0, :]

pt.multi_spectrum(
    df, 
    bins=bins, 
    suptitle='Detector Spectrums', 
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    output_folder='output/',
    linestyles = ['dotted', 'dotted', 'dotted', 'solid'],
    colors = ['blue', 'orange', 'green', 'black'],
    filename='DetectorSpectrumsCF'
)

import scipy.optimize as opt

def linear_combination(x, *params):
    return sum(p * xi for p, xi in zip(params, x))

def residuals(params, x, y):
    return y - linear_combination(x, *params)

def find_linear_combination(x, y, initial_guess=None):
    if initial_guess is None:
        initial_guess = np.array([1]*x.shape[0]) / x.shape[0]
    popt, _ = opt.leastsq(residuals, x0=initial_guess, args=(x, y))
    return popt


# assuming tnm_c50si50 is made up of 50% tnm_c and 50% tnm_si, prove this by finding the linear combination of the two using regression
# find the linear combination of tnm_c and tnm_si that best approximates tnm_c50si50
x = df[['tnm_c', 'tnm_si']].values.T
y = df['tnm_c50si50'].values
a, b = find_linear_combination(x, y)
a, b
print(f'Linear combination of tnm_c and tnm_si: {a} * tnm_c + {b} * tnm_si')

_df = df.copy()
_df['tnm_c'] = _df['tnm_c'] * a
_df['tnm_si'] = _df['tnm_si'] * b
_df['fit'] = _df['tnm_c'] + _df['tnm_si']

print(df.columns)
# _df['tnm_c50si50'] = _df['tnm_c50si50'] / _df['tnm_c50si50'].max()
# _df['tnm_c'] = _df['tnm_c'] / _df['tnm_c'].max()
# _df['tnm_si'] = _df['tnm_si'] / _df['tnm_si'].max()
# _df['tnm_c50si50'] = _df['tnm_c50si50'] / _df['tnm_c50si50'].max()

pt.multi_spectrum(
    _df[['tnm_c', 'tnm_si', 'tnm_c50si50', 'fit']], 
    bins=bins, 
    suptitle='Linear Combination of tnm_c and tnm_si', 
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    output_folder='output/',
    linestyles = ['dotted', 'dotted', 'solid', 'dashed'],
    colors = ['blue', 'orange', 'black', 'red'],
    filename='LinearCombinationTNMCF'
)

x = df[['tnm_c', 'tnm_si', 'tnm_al203']].values.T
y = df['tnm_c50si50']
a, b, c = find_linear_combination(x, y)
print(f'{a} * C + {b} * Si + {c} * Al2O3 = C/Si mix')
_df['tnm_c'] = _df['tnm_c'] * a
_df['tnm_si'] = _df['tnm_si'] * b
_df['tnm_al203'] = _df['tnm_al203'] * c
_df['fit'] = _df['tnm_c'] + _df['tnm_si'] + _df['tnm_al203']
# print(df.columns)
# _df['tnm_c50si50'] = _df['tnm_c50si50'] / _df['tnm_c50si50'].max()
# _df['tnm_c'] = _df['tnm_c'] / _df['tnm_c'].max()
# _df['tnm_si'] = _df['tnm_si'] / _df['tnm_si'].max()
# _df['tnm_al203'] = _df['tnm_al203'] / _df['tnm_al203'].max()
# _df['tnm_c50si50'] = _df['tnm_c50si50'] / _df['tnm_c50si50'].max()
pt.multi_spectrum(
    _df[['tnm_c', 'tnm_si', 'tnm_al203', 'tnm_c50si50', 'fit']], 
    bins=bins, 
    suptitle='Linear Combination of tnm_c, tnm_si and tnm_al203', 
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    output_folder='output/',
    linestyles = ['dotted', 'dotted', 'dotted', 'solid', 'dashed'],
    colors = ['blue', 'orange', 'green', 'black', 'red'],
    filename='LinearCombination2TNMCF'
)

# limit the data to the range of 4.2 to 4.7
_df_1 = df[(bins >= 4.2) & (bins <= 4.7)]
_df_2 = df[(bins >= 1.6) & (bins <= 1.95)]
_bins_1 = bins[(bins >= 4.2) & (bins <= 4.7)]
_bins_2 = bins[(bins >= 1.6) & (bins <= 1.95)]
_bins = np.concatenate((_bins_1, _bins_2))
_df = pd.concat([_df_1, _df_2], axis=0)
_df = _df.drop_duplicates()
_df = _df.reset_index(drop=True)

x = _df[['tnm_c', 'tnm_si', 'tnm_al203']].values.T
y = _df['tnm_c50si50']
a, b, c = find_linear_combination(x, y)
print(f'{a} * C + {b} * Si + {c} * Al2O3 = C/Si mix')
_df = df.copy()
_df['tnm_c'] = _df['tnm_c'] * a
_df['tnm_si'] = _df['tnm_si'] * b
_df['tnm_al203'] = _df['tnm_al203'] * c
_df['fit'] = _df['tnm_c'] + _df['tnm_si'] + _df['tnm_al203']
# print(df.columns)
# _df['tnm_c50si50'] = _df['tnm_c50si50'] / _df['tnm_c50si50'].max()
# _df['tnm_c'] = _df['tnm_c'] / _df['tnm_c'].max()
# _df['tnm_si'] = _df['tnm_si'] / _df['tnm_si'].max()
# _df['tnm_al203'] = _df['tnm_al203'] / _df['tnm_al203'].max()
# _df['tnm_c50si50'] = _df['tnm_c50si50'] / _df['tnm_c50si50'].max()
pt.multi_spectrum(
    _df[['tnm_c', 'tnm_si', 'tnm_al203', 'tnm_c50si50', 'fit']], 
    bins=bins, 
    suptitle='Linear Combination of tnm_c, tnm_si and tnm_al203', 
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    output_folder='output/',
    linestyles = ['dotted', 'dotted', 'dotted', 'solid', 'dashed'],
    colors = ['blue', 'orange', 'green', 'black', 'red'],
    filename='LinearCombination3TNMCF'
)