# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %%
output_folder = "data/output_data/"

# %%
# c window and si window are used in fitting, low window is used in plotting
c_window = [460, 530] 
si_window = [175, 220]
low_window = [0, 100]
# default peak max height values, auto_peak_max will set the max height to the max value in the window
c_peak_max = 5
si_peak_max = 5
auto_peak_max = True

# %%
# to read xlsx in "data/input_data/Combined Detectors dens_0_6.xlsx"
data_folder = "data/input_data/"
file_name = "Combined Detectors_dens_2_0"
df = pd.read_excel(data_folder + file_name+'.xlsx')
df = df[7:]
df = df[df.columns[3:]]
df.columns = df.iloc[1]
df = df[2:]
df = df.reset_index(drop=True)
bins = np.arange(0, len(df), 1)
df["bin"] = bins
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

# %%
# this is to be able to input a list of values for the spectrum
# df = pd.DataFrame()
# spec = []
# bins = np.arange(0, len(spec), 1)
# df['bin'] = bins
# df['spec'] = spec
# file_name = 'test'
# bins = np.arange(0, len(df), 1)
# df["bin"] = bins



# %%
def plott(df, suptitle):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(suptitle)

    c_maxs = []
    c_mins = []
    for col in df.columns[1:]:
        spec = df[col]
        axs[0, 1].plot(bins, spec, label=col, marker='o')
        filtered = spec[(bins > c_window[0]) & (bins < c_window[1])]
        c_maxs.append(np.max(filtered))
        c_mins.append(np.min(filtered))
    axs[0, 1].legend()
    axs[0, 1].set_title('Carbon Window (Zoomed)')
    axs[0, 1].set_xlabel('MeV')
    axs[0, 1].set_ylabel('Intensity')
    axs[0, 1].set_xlim(c_window[0], c_window[1])
    axs[0, 1].set_ylim(np.min(c_mins), np.max(c_maxs))

    # Third subplot
    si_maxs = []
    si_mins = []
    for col in df.columns[1:]:
        spec = df[col]
        axs[1, 0].plot(bins, spec, label=col, marker='o')
        filtered = spec[(bins > si_window[0]) & (bins < si_window[1])]
        si_maxs.append(np.max(filtered))
        si_mins.append(np.min(filtered))
    # axs[1, 0].legend()
    axs[1, 0].set_title('Silicone Window (Zoomed)')
    axs[1, 0].set_xlabel('MeV')
    axs[1, 0].set_ylabel('Intensity')
    axs[1, 0].set_xlim(si_window[0], si_window[1])
    axs[1, 0].set_ylim(np.min(si_mins), np.max(si_maxs))

    # Fourth subplot
    low_maxs = []
    low_mins = []
    for col in df.columns[1:]:
        spec = df[col]
        axs[1, 1].plot(bins, spec, label=col, marker='o')
        filtered = spec[(bins > low_window[0]) & (bins < low_window[1])]
        low_maxs.append(np.max(filtered))
        low_mins.append(np.min(filtered))
    axs[1, 1].legend()
    axs[1, 1].set_title('Low Energy (Zoomed)')
    axs[1, 1].set_xlabel('MeV')
    axs[1, 1].set_ylabel('Intensity')
    axs[1, 1].set_xlim(low_window[0], low_window[1])
    axs[1, 1].set_ylim(np.min(low_mins), np.max(low_maxs))

    # First subplot
    for col in df.columns[1:]:
        spec = df[col]
        axs[0, 0].plot(bins, spec, label=col)
    # axs[0, 0].legend()
    axs[0, 0].set_title('Spectrums')
    axs[0, 0].set_xlabel('MeV')
    axs[0, 0].set_ylabel('Intensity')
    # # draw squares around the zoomed regions
    for ax in axs:
        for a in ax:
            a.add_patch(plt.Rectangle((low_window[0], np.min(low_mins)), low_window[1]-low_window[0], np.max(low_maxs), fill=None, edgecolor='red'))
            a.add_patch(plt.Rectangle((si_window[0], np.min(si_mins)), si_window[1]-si_window[0], np.max(si_maxs), fill=None, edgecolor='red'))
            a.add_patch(plt.Rectangle((c_window[0], np.min(c_mins)), c_window[1]-c_window[0], np.max(c_maxs), fill=None, edgecolor='red'))
    # label the zoomed regions
    axs[0, 0].text(low_window[0], np.max(low_maxs), 'Low Energy', horizontalalignment='center', verticalalignment='center', transform=axs[0, 0].transData, color='red')
    axs[0, 0].text(c_window[0], np.max(c_maxs), 'Carbon', horizontalalignment='center', verticalalignment='center', transform=axs[0, 0].transData, color='red')
    axs[0, 0].text(si_window[0], np.max(si_maxs), 'Silicone', horizontalalignment='center', verticalalignment='center', transform=axs[0, 0].transData, color='red')

    # # log_toggle = [True, True, True, True]
    log_toggle = [True, False, False, False]

    for i in range(log_toggle.__len__()):
        if log_toggle[i]:
            axs[i//2, i%2].set_yscale('log')

    # log scale
    plt.tight_layout()
    # plt.show()

    plt.savefig(f"{output_folder+suptitle}.jpg")


# %%
plott(df, file_name+' Spectrums')

# %%
def linearfunc(x, a, b):
    return a * x + b

def gaussianfunc(x, a, b, c):
    return a * np.exp(-((x - b) / c) ** 2)

def geb(x, a, b, c):
    return (a+b*np.sqrt(x+c*(x*x)))*0.60056120439322


def exp_falloff(x,x0,a,p,b):
    return (a*np.exp(-p*(x-x0)))+b

# %%
# auto peak max
if auto_peak_max:
    c_df_window = df[df.columns[1:]].values[c_window[0]:c_window[1]]
    si_df_window = df[df.columns[1:]].values[si_window[0]:si_window[1]]
    c_peak_max = np.max(df[df.columns[1:]].values)- np.min(df[df.columns[1:]].values)
    si_peak_max = np.max(df[df.columns[1:]].values) - np.min(df[df.columns[1:]].values)

# %%
# the following functions are used in curve_fit, 
# the baseline can be any function as long as the bounds and initial params are the correct size, 
# to switch the peak function, also change the area under peak function

si_baseline = lambda x, a, b: linearfunc(x, a, b)
si_baseline_p0 = [0, 0] # a, b
si_basline_lower = [-np.inf, -np.inf]
si_basline_upper = [np.inf, np.inf]

# with exp_falloff
# si_baseline = lambda x, a, b, c, d: exp_falloff(x, a, b, c, d)
# si_baseline_p0 = [0, 0, 0, 0] # a, b, c, d
# si_basline_lower = [-np.inf, -np.inf, -np.inf, -np.inf]
# si_basline_upper = [np.inf, np.inf, np.inf, np.inf]

si_peak = lambda x, a, b, c: gaussianfunc(x, a, b, c)
si_peak_p0 = [si_peak_max/2, (si_window[1]+si_window[0])/2, 5]
si_peak_lower = [0, si_window[0], 1]
si_peak_upper = [si_peak_max, si_window[1], 50]

c_baseline = lambda x, a, b: linearfunc(x, a, b)
c_baseline_p0 = [0, 0]
c_basline_lower = [-np.inf, -np.inf]
c_basline_upper = [np.inf, np.inf]

c_peak = lambda x, a, b, c: gaussianfunc(x, a, b, c)
c_peak_p0 = [c_peak_max, (c_window[1]+c_window[0])/2, 2] # staring params
c_peak_lower = [0, c_window[0], 1]
c_peak_upper = [c_peak_max, c_window[1], 50]

# Double Peak
# c_peak = lambda x, a, b, c, d, e, f: gaussianfunc(x, a, b, c) + gaussianfunc(x, d, e, f)
# c_peak_p0 = [c_peak_max, (c_window[1]+c_window[0])/2, 2]*2 # staring params
# c_peak_lower = [0, c_window[0], 1]*2
# c_peak_upper = [c_peak_max*2, c_window[1], 5]*2




# %%
c_p0 = c_baseline_p0 + c_peak_p0
c_lower = c_basline_lower + c_peak_lower
c_upper = c_basline_upper + c_peak_upper
c_bounds = (c_lower, c_upper)

si_p0 = si_baseline_p0 + si_peak_p0
si_lower = si_basline_lower + si_peak_lower
si_upper = si_basline_upper + si_peak_upper
si_bounds = (si_lower, si_upper)

def total_c(x, *params):
    return c_baseline(x, *params[:len(c_baseline_p0)]) + c_peak(x, *params[len(c_baseline_p0):])

def total_si(x, *params):
    return si_baseline(x, *params[:len(si_baseline_p0)]) + si_peak(x, *params[len(si_baseline_p0):])




# %%
def area_under_gaussian_peak(a, b, c):
    area = a * c * np.sqrt(np.pi)
    return area

def generic_area_under_peak(peak_func, x, *params):
    area = np.trapezoid(peak_func(x, *params), x)
    return area

# %%
# carbon fitting
c_popts = []
c_pcovs = []
c_infodicts = []
c_mesgs = []
c_iers = []
c_peak_areas = []
c_peak_fitting_errs = []

for col in df.columns[1:]:
    x_fit = bins[c_window[0]:c_window[1]]
    y_fit = df[col][c_window[0]:c_window[1]]
    p0 = np.clip(c_p0, c_lower, c_upper)
    popt, pcov, infodict, mesg, ier = curve_fit(
        total_c, 
        x_fit, 
        y_fit, 
        p0=p0,
        bounds=c_bounds,
        full_output=True
    )
    c_popts.append(popt)
    c_pcovs.append(pcov)
    c_infodicts.append(infodict)
    c_mesgs.append(mesg)
    c_iers.append(ier)

    # c_peak_area = area_under_gaussian_peak(*popt[len(c_baseline_p0):]) # will need to change this if the peak function changes
    c_peak_area = generic_area_under_peak(c_peak, x_fit, *popt[len(c_baseline_p0):])
    c_peak_areas.append(c_peak_area)

    c_peak_fitting_err = (y_fit - total_c(x_fit, *popt)).std()
    c_peak_fitting_errs.append(c_peak_fitting_err)

# silicon fitting    
si_popts = []
si_pcovs = []
si_infodicts = []
si_mesgs = []
si_iers = []
si_peak_areas = []
si_peak_fitting_errs = []

for col in df.columns[1:]:
    x_fit = bins[si_window[0]:si_window[1]]
    y_fit = df[col][si_window[0]:si_window[1]]
    # Ensure initial guess p0 is within bounds
    p0 = np.clip(si_p0, si_lower, si_upper)
    popt, pcov, infodict, mesg, ier = curve_fit(
        total_si, 
        x_fit, 
        y_fit, 
        p0=p0,
        bounds=si_bounds,
        full_output=True
    )
    si_popts.append(popt)
    si_pcovs.append(pcov)
    si_infodicts.append(infodict)
    si_mesgs.append(mesg)
    si_iers.append(ier)

    # si_peak_area = area_under_gaussian_peak(*popt[len(si_baseline_p0):]) # will need to change this if the peak function changes
    si_peak_area = generic_area_under_peak(si_peak, x_fit, *popt[len(si_baseline_p0):])
    si_peak_areas.append(si_peak_area)

    si_peak_fitting_err = (y_fit - total_si(x_fit, *popt)).std()
    si_peak_fitting_errs.append(si_peak_fitting_err)

# %%
# print out the results
for i, col in enumerate(df.columns[1:]):
    print("**************")
    print(f"Mo: {col}")
    print(f"Carbon Peak Area: {c_peak_areas[i]}")
    print(f"Silicone Peak Area: {si_peak_areas[i]}")
    print(f"Carbon Peak Fitting Error: {c_peak_fitting_errs[i]}")
    print(f"Silicone Peak Fitting Error: {si_peak_fitting_errs[i]}")
    print("Peak Parameters")
    print(f"Carbon Baseline: {c_popts[i][:len(c_baseline_p0)]}")
    print(f"Carbon Peak: {c_popts[i][len(c_baseline_p0):]}")
    print(f"Silicone Baseline: {si_popts[i][:len(si_baseline_p0)]}")
    print(f"Silicone Peak: {si_popts[i][len(si_baseline_p0):]}")
    print("**************")

# %%
fitting_df = pd.DataFrame()
fitting_df["Mo"] = df.columns[1:]
fitting_df["Carbon Peak Area"] = c_peak_areas
fitting_df["Carbon Peak Area Error"] = c_peak_fitting_errs
fitting_df["Silicone Peak Area"] = si_peak_areas
fitting_df["Silicone Peak Area Error"] = si_peak_fitting_errs

# %%
fitting_df.to_excel(output_folder + file_name + " Fitting Results.xlsx")

# %%
# plot the fitted data
fig, axs = plt.subplots(len(df.columns[1:]), 2, figsize=(10, 5*len(df.columns[1:])))
axs = np.atleast_2d(axs)
suptitle = "Fitted Data"
plt.suptitle(file_name+' '+suptitle, fontsize=16, fontweight='bold', y=1)

si_mins = []
si_maxs = []

c_mins = []
c_maxs = []

for i, col in enumerate(df.columns[1:]):
    
    x = bins
    y = df[col]
    axs[i, 0].plot(x, y, label=col)
    c_mins.append(np.min(y[c_window[0]:c_window[1]]))
    c_maxs.append(np.max(y[c_window[0]:c_window[1]]))
    
    fit = total_c(x, *c_popts[i])[c_window[0]:c_window[1]]
    c_mins.append(np.min(fit))
    c_maxs.append(np.max(fit))
    baseline_fit = c_baseline(x, *c_popts[i][:len(c_baseline_p0)])[c_window[0]:c_window[1]]
    axs[i, 0].plot(x[c_window[0]:c_window[1]], baseline_fit, label=str(col)+'_baseline', linestyle='--')
    axs[i, 0].plot(x[c_window[0]:c_window[1]], fit, label=str(col)+'_fit', linestyle='--')
    axs[i, 0].set_title('Fitted Carbon Window')
    axs[i, 0].set_xlabel('MeV')
    axs[i, 0].set_ylabel('Intensity')
    c_mins.append(np.min(total_c(x[c_window[0]], *c_popts[i])))
    axs[i, 0].legend()

    x = bins
    y = df[col]
    axs[i, 1].plot(x, y, label=col)
    si_mins.append(np.min(y[si_window[0]:si_window[1]]))
    si_maxs.append(np.max(y[si_window[0]:si_window[1]]))
    
    fit = total_si(x, *si_popts[i])[si_window[0]:si_window[1]]
    si_mins.append(np.min(fit))
    si_maxs.append(np.max(fit))
    baseline_fit = si_baseline(x, *si_popts[i][:len(si_baseline_p0)])[si_window[0]:si_window[1]]
    axs[i, 1].plot(x[si_window[0]:si_window[1]], baseline_fit, label=str(col)+'_baseline', linestyle='--')
    axs[i, 1].plot(x[si_window[0]:si_window[1]], fit, label=str(col)+'_fit', linestyle='--')
    axs[i, 1].set_title('Fitted Silicone Window')
    axs[i, 1].set_xlabel('MeV')
    axs[i, 1].set_ylabel('Intensity')
    axs[i, 1].legend()


# # set mins and maxs for the y axis
[axs[i, 0].set_ylim(np.min(c_mins), np.max(c_maxs)) for i in range(axs.shape[0])]
[axs[i, 1].set_ylim(np.min(si_mins), np.max(si_maxs)) for i in range(axs.shape[0])]
# # set mins and maxs for the x axis
[axs[i, 0].set_xlim(c_window[0], c_window[1]) for i in range(axs.shape[0])]
[axs[i, 1].set_xlim(si_window[0], si_window[1]) for i in range(axs.shape[0])]


plt.tight_layout()
plt.savefig(f"{output_folder+file_name+' '+suptitle}.jpg")

# %%
carbon_fitting_df = pd.DataFrame()
carbon_fitting_df["Mo"] = df.columns[1:]
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i in range(len(c_baseline_p0)):
    np.array(c_popts).shape
    carbon_fitting_df[f"Baseline {alphabet[i]}"] = np.array(c_popts)[:, i]
for i in range(len(c_peak_p0)):
    carbon_fitting_df[f"Peak {alphabet[i]}"] = np.array(c_popts)[:, i+len(c_baseline_p0)]

carbon_fitting_df.to_excel(output_folder + file_name + " Carbon Fitting Results.xlsx")

# %%
si_fitting_df = pd.DataFrame()
si_fitting_df["Mo"] = df.columns[1:]
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i in range(len(si_baseline_p0)):
    np.array(si_popts).shape
    si_fitting_df[f"Baseline {alphabet[i]}"] = np.array(si_popts)[:, i]
for i in range(len(si_peak_p0)):
    si_fitting_df[f"Peak {alphabet[i]}"] = np.array(si_popts)[:, i+len(si_baseline_p0)]

si_fitting_df.to_excel(output_folder + file_name + " Silicone Fitting Results.xlsx")

# %%
# make a dataframe of bins, true, baseline and fitted data on carbon window
carbon_fitting_df = pd.DataFrame()
filtered_bins = bins[c_window[0]:c_window[1]]
carbon_fitting_df["bin"] = filtered_bins

for i, col in enumerate(df.columns[1:]):
    carbon_fitting_df[str(col)+'_baseline'] = c_baseline(filtered_bins, *c_popts[i][:len(c_baseline_p0)])
    carbon_fitting_df[str(col)+'_fit'] = total_c(filtered_bins, *c_popts[i])

# %%
silicone_fitting_df = pd.DataFrame()
filtered_bins = bins[si_window[0]:si_window[1]]
silicone_fitting_df["bin"] = filtered_bins

for i, col in enumerate(df.columns[1:]):
    silicone_fitting_df[str(col)+'_baseline'] = si_baseline(filtered_bins, *si_popts[i][:len(si_baseline_p0)])
    silicone_fitting_df[str(col)+'_fit'] = total_si(filtered_bins, *si_popts[i])