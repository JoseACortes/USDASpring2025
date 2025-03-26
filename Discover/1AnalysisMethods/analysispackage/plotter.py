import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def general_plotter(df, bins=None, suptitle=None, c_window=None, si_window=None, low_window=None, output_folder=None):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(suptitle)

        c_maxs = []
        c_mins = []
        for col in df.columns:
            spec = df[col]
            axs[0, 1].plot(bins, spec, label=col, marker='o')
            filtered = spec[(bins >= c_window[0]) & (bins <= c_window[1])]
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
        for col in df.columns:
            spec = df[col]
            axs[1, 0].plot(bins, spec, label=col, marker='o')
            filtered = spec[(bins >= si_window[0]) & (bins <= si_window[1])]
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
        for col in df.columns:
            spec = df[col]
            axs[1, 1].plot(bins, spec, label=col, marker='o')
            filtered = spec[(bins >= low_window[0]) & (bins <= low_window[1])]
            low_maxs.append(np.max(filtered))
            low_mins.append(np.min(filtered))
        axs[1, 1].legend()
        axs[1, 1].set_title('Low Energy (Zoomed)')
        axs[1, 1].set_xlabel('MeV')
        axs[1, 1].set_ylabel('Intensity')
        axs[1, 1].set_xlim(low_window[0], low_window[1])
        axs[1, 1].set_ylim(np.min(low_mins), np.max(low_maxs))

        # First subplot
        for col in df.columns:
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
        plt.close(fig)

def carbon_plotter(df, bins=None, suptitle=None, c_window=None, output_folder=None, filetype='jpg'):
        fig, axs = plt.subplots(1, 1, figsize=(15, 10), frameon=False)
        fig.suptitle(suptitle)
        c_maxs = []
        c_mins = []
        for col in df.columns:
            spec = df[col]
            axs.plot(bins, spec, label=col, marker='o')
            filtered = spec[(bins >= c_window[0]) & (bins <= c_window[1])]
            c_maxs.append(np.max(filtered))
            c_mins.append(np.min(filtered))
        axs.legend()
        axs.set_title('Carbon Window (Zoomed)')
        axs.set_xlabel('MeV')
        axs.set_ylabel('Intensity')
        axs.set_xlim(c_window[0], c_window[1])
        axs.set_ylim(np.min(c_mins), np.max(c_maxs))
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{output_folder+suptitle}."+filetype)
        plt.close(fig)
    
def single_carbon_fit_plotter(df, column=None, bins=None, c_window=None, suptitle=None, output_folder=None, filetype='jpg'):
    """
    Generates and saves a single carbon spectrum plot based on the provided data.
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the spectrum data.
        column (str or int, optional): The column name or index in the DataFrame to plot. 
                                       Defaults to the first column if not specified.
        bins (numpy.ndarray): The array of bin edges corresponding to the spectrum data.
        c_window (tuple): A tuple specifying the range (min, max) of bins to include in the plot.
        suptitle (str, optional): The title of the plot. This will also be used as part of the output filename.
        output_folder (str, optional): The folder where the plot will be saved. 
                                       The full path will be `output_folder + suptitle`.
        filetype (str, optional): The file format for saving the plot (e.g., 'jpg', 'png'). Defaults to 'jpg'.
    Returns:
        None: The function saves the plot to the specified location and does not return any value.
    Notes:
        - The y-axis is set to a logarithmic scale.
        - The plot is saved with a resolution of 300 DPI.
        - The function assumes that `bins` and `df[column]` are aligned in terms of indexing.
        - The function requires `numpy` and `matplotlib.pyplot` to be imported as `np` and `plt`, respectively.
    """
    if column is None:
        column = df.columns[0:3]
    if isinstance(column, int):
        column = df.columns[3*column:3*column+3]

    c_filter = (bins >= c_window[0]) & (bins <= c_window[1])

    c_bins = bins[c_filter]

    c_spec = df[column][c_filter]

    c_min, c_max = np.min(c_spec), np.max(c_spec)

    fig = plt.figure(figsize=(5.45, 5.59), dpi=300, frameon=False)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=.98)

    axs = fig.add_subplot(111)
    axs.set_title('MINS Readings')
    
    axs.plot(c_bins, c_spec[c_spec.columns[2]], label=c_spec.columns[2], color='black')
    axs.plot(c_bins, c_spec[c_spec.columns[0]], label=c_spec.columns[0], color='blue')
    axs.plot(c_bins, c_spec[c_spec.columns[1]], label=c_spec.columns[1], color='red')
    
    plt.fill_between(c_bins, c_spec[c_spec.columns[0]], c_spec[c_spec.columns[1]], color='red', alpha=0.1)

    # axs.set_yscale('log')
    axs.set_xlabel('Energy (MeV)')
    axs.set_ylabel('Counts')
    axs.grid()
    plt.legend()
    
    plt.savefig(f"{output_folder+suptitle}."+filetype)
    plt.close(fig)

def silicone_plotter(df, bins=None, suptitle=None, c_window=None, si_window=None, low_window=None, output_folder=None, filetype='jpg'):
        fig, axs = plt.subplots(1, 1, figsize=(15, 10), frameon=False)
        fig.suptitle(suptitle)
        si_maxs = []
        si_mins = []
        for col in df.columns:
            spec = df[col]
            axs.plot(bins, spec, label=col, marker='o')
            filtered = spec[(bins >= si_window[0]) & (bins <= si_window[1])]
            si_maxs.append(np.max(filtered))
            si_mins.append(np.min(filtered))
        axs.legend()
        axs.set_title('Silicone Window (Zoomed)')
        axs.set_xlabel('MeV')
        axs.set_ylabel('Intensity')
        axs.set_xlim(si_window[0], si_window[1])
        axs.set_ylim(np.min(si_mins), np.max(si_maxs))
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{output_folder+suptitle}."+filetype)
        plt.close(fig)

def single_silicone_plotter(df, column=None, bins=None, si_window=None, suptitle=None, output_folder=None, filetype='jpg'):
    if column is None:
        column = df.columns[0]
    if isinstance(column, int):
        column = df.columns[column]

    si_filter = (bins >= si_window[0]) & (bins <= si_window[1])

    si_bins = bins[si_filter]

    si_spec = df[column][si_filter]

    si_min, si_max = np.min(si_spec), np.max(si_spec)

    fig = plt.figure(figsize=(5.96, 8.67), dpi=300, frameon=False)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.95)

    axs = fig.add_subplot(111)
    axs.set_title('MINS Readings')
    
    axs.plot(si_bins, si_spec, label=column, color='black')
    axs.set_yscale('log')
    axs.set_xlabel('Energy (MeV)')
    axs.set_ylabel('Counts')
    plt.legend()
    
    plt.savefig(f"{output_folder+suptitle}."+filetype)
    plt.grid()
    plt.close(fig)

def fits_plotter(df, c_df, si_df, c_bins, si_bins, suptitle=None, output_folder=None):
        fig, axs = plt.subplots(len(df.columns), 2, figsize=(15, 10*len(df.columns)))
        fig.suptitle(suptitle)
        
        for i, col in enumerate(df.columns):
            true_spec = c_df[col+" true"]
            peak_spec = c_df[col+" peak"]
            baseline_spec = c_df[col+" baseline"]
            axs[i, 0].plot(c_bins, true_spec, label='True', marker='o')
            axs[i, 0].plot(c_bins, peak_spec, label='Peak', marker='o')
            axs[i, 0].plot(c_bins, baseline_spec, label='Baseline', marker='o')
            axs[i, 0].legend()
            axs[i, 0].set_title('Carbon Fits')
            axs[i, 0].set_xlabel('MeV')
            axs[i, 0].set_ylabel('Intensity')
            axs[i, 0].set_title(f'Carbon Fit {col}')
            
            true_spec = si_df[col+" true"]
            peak_spec = si_df[col+" peak"]
            baseline_spec = si_df[col+" baseline"]
            axs[i, 1].plot(si_bins, true_spec, label='True', marker='o')
            axs[i, 1].plot(si_bins, peak_spec, label='Peak', marker='o')
            axs[i, 1].plot(si_bins, baseline_spec, label='Baseline', marker='o')
            axs[i, 1].legend()
            axs[i, 1].set_title('Silicone Fits')
            axs[i, 1].set_xlabel('MeV')
            axs[i, 1].set_ylabel('Intensity')
            axs[i, 1].set_title(f'Silicone Fit {col}')
        # plt.show()
        plt.savefig(f"{output_folder+suptitle}.jpg")
        plt.close(fig)

def plot_fitting_results(fitting_df, true_peak_area, column = None, element='Carbon', suptitle=None, output_folder=None, filetype='jpg'):
    
    fig, axs = plt.subplots(1, 1, figsize=(5.45, 5.59), frameon=False)
    fig.suptitle(suptitle)
    axs.scatter(true_peak_area, fitting_df[element+ ' Peak Area'], label='Fitted Peak Area', marker='o', color='red')
    if column is not None:
        axs.plot(true_peak_area[column], fitting_df[element+ ' Peak Area'][column], marker='x', color='red')
    axs.legend()
    axs.set_title('Fitted Peak Area vs Element Concentration')
    axs.set_xlabel('True '+element+' %')
    axs.set_ylabel('Fitted '+element+' Peak Area')
    plt.grid()
    # plt.tight_layout()

    # plt.show()
    plt.savefig(f"{output_folder+suptitle}."+filetype)
    plt.close(fig)

def plot_reg_results(x_hat, true_peak_area, element='Carbon', training_mask = None, suptitle=None, output_folder=None):
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    fig.suptitle(suptitle)
    true_peak_area = np.array(true_peak_area)
    x_hat = np.array(x_hat)
    x_min = np.min([true_peak_area, x_hat], axis=-1)
    x_max = np.max([true_peak_area, x_hat], axis=-1)
    axs.plot([x_min, x_max], [x_min, x_max], color='black', linestyle='-.', alpha=0.1) 
    if training_mask is not None:
        axs.scatter(true_peak_area[training_mask], x_hat[training_mask], label='Training Data', marker='o')
        axs.scatter(true_peak_area[~training_mask], x_hat[~training_mask], label='Testing Data', marker='o')
    else:
        axs.scatter(true_peak_area, x_hat, label='All Data', marker='o')
    
    axs.legend()
    axs.set_title('Fitting Results')
    axs.set_xlabel('True '+element+' C%')
    axs.set_ylabel('Predicted '+element+' C%')
    plt.tight_layout()
    # square
    axs.set_aspect('equal', adjustable='box')
    # plt.show()
    plt.savefig(f"{output_folder+suptitle}.jpg")
    plt.close(fig)

def single_spectrum(df, column=None, bins=None, c_window=None, si_window=None, suptitle=None, output_folder=None, large_window=[.1, np.inf]):
    if column is None:
        column = df.columns[0]
    if isinstance(column, int):
        column = df.columns[column]

    
    si_filter = (bins >= si_window[0]) & (bins <= si_window[1])
    c_filter = (bins >= c_window[0]) & (bins <= c_window[1])
    
    si_bins = bins[si_filter]
    c_bins = bins[c_filter]
    
    si_spec = df[column][si_filter]
    c_spec = df[column][c_filter]

    si_min, si_max = np.min(si_spec), np.max(si_spec)
    c_min, c_max = np.min(c_spec), np.max(c_spec)

    large_window_filter = (bins >= large_window[0]) & (bins <= large_window[1])
    large_window_bins = bins[large_window_filter]
    large_window_spec = df[column][large_window_filter]
    fig, axs = plt.subplot_mosaic(
        """
        AA
        BC
        """, figsize=(5.96, 8.67), dpi=300, frameon=False)
    
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.85)

    axs['A'].set_title('MINS Readings')
    axs['A'].plot(large_window_bins, 
                  large_window_spec, label='Spectrum', color='black')
    axs['A'].set_yscale('log')
    axs['A'].set_xlabel('Energy (MeV)')
    axs['A'].set_ylabel('Counts')
    # draw squares around the zoomed regions
    axs['A'].add_patch(plt.Rectangle((c_window[0], c_min), c_window[1]-c_window[0], c_max, fill=None, edgecolor='red'))
    axs['A'].add_patch(plt.Rectangle((si_window[0], si_min), si_window[1]-si_window[0], si_max, fill=None, edgecolor='red'))
    # label the zoomed regions
    axs['A'].text((c_window[1]+c_window[0])/2, c_max*1.5, 'Carbon', horizontalalignment='center', verticalalignment='bottom', transform=axs['A'].transData, color='red')
    axs['A'].text((si_window[1]+si_window[0])/2, si_max*1.5, 'Silicone', horizontalalignment='center', verticalalignment='bottom', transform=axs['A'].transData, color='red')

    axs['B'].plot(si_bins, si_spec, label='Spectrum', color='black') 
    axs['B'].set_xlim(si_window[0], si_window[1])
    # axs['B'].set_ylim(si_min, si_max)
    axs['B'].set_title('Silicone Peak')
    axs['B'].set_xlabel('Energy (MeV)')
    axs['B'].set_ylabel('Counts')

    axs['C'].plot(c_bins, c_spec, label='Spectrum', color='black')
    axs['C'].set_xlim(c_window[0], c_window[1])
    # axs['C'].set_ylim(c_min, c_max)
    axs['C'].set_title('Carbon Peak')
    axs['C'].set_xlabel('Energy (MeV)')
    axs['C'].set_ylabel('Counts')

    start = axs['A'].transData.transform((c_window[0], c_min))
    end = axs['C'].transData.transform((c_window[0], axs['C'].get_ylim()[1]))
    # Transform figure coordinates to display space
    inv = fig.transFigure.inverted()
    start_fig = inv.transform(start)
    end_fig = inv.transform(end)
    # Add the line in figure space
    line = Line2D(
        [start_fig[0], end_fig[0]],  # x-coordinates in figure space
        [start_fig[1], end_fig[1]],  # y-coordinates in figure space
        transform=fig.transFigure,  # Use figure transformation
        color='blue', linestyle='--', linewidth=1
    )
    fig.add_artist(line)

    start = axs['A'].transData.transform((c_window[1], c_min))
    end = axs['C'].transData.transform((c_window[1], axs['C'].get_ylim()[1]))
    # Transform figure coordinates to display space
    inv = fig.transFigure.inverted()
    start_fig = inv.transform(start)
    end_fig = inv.transform(end)
    # Add the line in figure space
    line = Line2D(
        [start_fig[0], end_fig[0]],  # x-coordinates in figure space
        [start_fig[1], end_fig[1]],  # y-coordinates in figure space
        transform=fig.transFigure,  # Use figure transformation
        color='blue', linestyle='--', linewidth=1
    )
    fig.add_artist(line)


    # plt.tight_layout()
    plt.savefig(f"{output_folder+suptitle}.png")
    plt.close(fig)

def single_component_fit(df, testing_df, fitting_df, bins=None, window=None, column=None, suptitle=None, output_folder=None, filetype='jpg'):
    """
    Perform a single-component fit on the provided data and generate a plot of the results.
    Parameters:
    -----------
    df : pandas.DataFrame
        The main DataFrame containing the true data to be plotted.
    testing_df : pandas.DataFrame
        The DataFrame containing the testing data to be compared against the true data.
    fitting_df : pandas.DataFrame
        The DataFrame containing the fitting weights for each component.
    bins : array-like, optional
        The bin edges for the histogram. If None, defaults to the range of the first column in `df`.
    column : str or int, optional
        The column name or index in `df` to be used for the fit. If None, defaults to the first column.
    suptitle : str, optional
        The title for the plot. If None, no title is added.
    output_folder : str, optional
        The folder path where the plot will be saved. If None, the plot is not saved.
    filetype : str, optional
        The file type for saving the plot (e.g., 'jpg', 'png'). Defaults to 'jpg'.
    Returns:
    --------
    None
        The function saves the plot to the specified output folder and file type.
    Notes:
    ------
    - The function generates a plot with three lines: the true data, the testing data, and the fitted spectrum.
    - The plot includes a legend, axis labels, and a grid for better visualization.
    - The function does not display the plot but saves it to the specified location.
    """
    if column is None:
        column = df.columns[0]
    if isinstance(column, int):
        column = df.columns[column]
    
    if bins is None:
        bins = np.arange(len(df[column]))

    tst_clms = testing_df.columns
    spectrum_fit = [weight*testing_df[clm] for clm, weight in zip(tst_clms, fitting_df.loc[column])]
    spectrum_fit = np.sum(spectrum_fit, axis=0)
    fig, axs = plt.subplots(1, 1, figsize=(5.45, 5.59), dpi=300, frameon=False)
    fig.suptitle(suptitle)
    true_spec = df[column]
    if window is not None:
        c_filter = (bins >= window[0]) & (bins <= window[1])
        bins = bins[c_filter]
        true_spec = true_spec[c_filter]
        testing_df = testing_df[c_filter]
        spectrum_fit = spectrum_fit[c_filter]
    axs.plot(bins, true_spec, label='True '+column, color='black')
    for col in testing_df.columns:
        axs.plot(bins, testing_df[col], label='Train: '+col, color='blue')
    # axs.plot(bins, testing_df, label='Testing', color='blue')
    axs.plot(bins, spectrum_fit, label='Train Lin. Comb.', color='red')
    axs.legend()
    axs.set_title('Fitting Results')
    axs.set_xlabel('Energy (MeV)')
    axs.set_ylabel('Intensity')
    plt.grid()
    # plt.tight_layout()

    # plt.show()
    plt.savefig(f"{output_folder+suptitle}."+filetype)
    plt.close(fig)
    