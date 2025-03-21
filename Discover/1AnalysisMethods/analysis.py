import pandas as pd
import numpy as np

from analysispackage import plotter as pt

# Import
import pandas as pd
import numpy as np
import INS_Analysis as insd
import seaborn as sns
import matplotlib.pyplot as plt

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
headers = [str(x[0])[2:][:2]+'.'+str(x[0])[2:][2:4]+"%C" for x in concentrations]
df = pd.DataFrame(spectrums.T, columns=headers)

# In[2]: Plotting

