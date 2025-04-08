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


pt.multi_spectrum(
    df, 
    bins=bins, 
    columns=[df.columns[8]],
    suptitle='Carbon and Silicone Concentrations Spectrum', 
    c_window=[4.2, 4.7], 
    si_window=[1.6, 1.95],
    output_folder='output/',
    filename='carbon_silicone_concentrations_single.png',
    colors=['black']
    )