#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from mcnptools import Mctal, MctalTally
import INS_Analysis as insd
import numpy as np
import pandas as pd


# # Generate Index

# In[2]:


trials_file = 'inp/filenames.csv'
trials = pd.read_csv(trials_file)


# In[3]:


file_prefix = '../compute/output/mctal/'
file_type = '.mctal'
# check if the file exists
filenames = []
for index, row in trials.iterrows():
    # print('loading', row['name'])
    filename = file_prefix + row['name'] + file_type
    # check if the file exists
    if os.path.isfile(filename):
        filenames.append(row['name'])
filenames

trials = trials[trials['name'].isin([os.path.basename(f) for f in filenames])]
trials['filename'] = file_prefix + trials['name'] + file_type
trials = trials.reset_index(drop=True)

trials['ptrac_filename'] = '../compute/output/ptrac/' + trials['name'] + '.ptrac'

# first_n = 10
# trials = trials.head(first_n)
trials.reset_index(drop=False, inplace=True)
print('n-trials: ', len(trials))


# In[4]:


trials.to_csv('trials.csv', index=False)


# # Numpy

# In[8]:


# progress bar for the loop
from tqdm import tqdm
tqdm.pandas()


# In[9]:


spectrograms = []
bins = None

for filename in tqdm(trials['filename']):
    bins, vals = insd.read(filename, tally=88, start_time_bin=0, end_time_bin=150, nps=1e7)
    spectrograms.append(np.array(vals))

spectrograms = np.array(spectrograms)
bins = np.array(bins)


# In[10]:


np.savez('spectrograms.npz', x=bins, y=spectrograms)


# In[11]:


# load the data
data = np.load('spectrograms.npz')


# In[12]:


spectrums = []
for filename in tqdm(trials['filename']):
    bins, vals = insd.read(filename, tally=58, start_time_bin=0, end_time_bin=3, nps=1e7)
    
    spectrums.append(np.array(vals))

spectrums = np.array(spectrums)


# In[13]:


np.savez('spectrums.npz', x=bins, y=spectrums)


# In[14]:


gebless_spectrums = []
for filename in tqdm(trials['filename']):
    bins, vals = insd.read(filename, tally=18, start_time_bin=0, end_time_bin=3, nps=1e7)
    
    gebless_spectrums.append(np.array(vals))

gebless_spectrums = np.array(gebless_spectrums)
bins = np.array(bins)


# In[15]:


np.savez('gebless_spectrums.npz', x=bins, y=gebless_spectrums)



