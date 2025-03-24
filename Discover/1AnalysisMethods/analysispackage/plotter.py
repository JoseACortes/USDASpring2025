import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def carbon_plotter(df, bins=None, suptitle=None, c_window=None, si_window=None, low_window=None, output_folder=None):
        fig, axs = plt.subplots(1, 1, figsize=(15, 10))
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
        plt.savefig(f"{output_folder+suptitle}.jpg")
        plt.close(fig)

def silicone_plotter(df, bins=None, suptitle=None, c_window=None, si_window=None, low_window=None, output_folder=None):
        fig, axs = plt.subplots(1, 1, figsize=(15, 10))
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
        plt.savefig(f"{output_folder+suptitle}.jpg")
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

def plot_fitting_results(fitting_df, true_peak_area, element='Carbon', suptitle=None, output_folder=None):
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    fig.suptitle(suptitle)
    axs.scatter(true_peak_area, fitting_df[element+ ' Peak Area'], label='Fitted Peak Area', marker='o')
    axs.legend()
    axs.set_title('Fitting Results')
    axs.set_xlabel('MeV')
    axs.set_ylabel('Intensity')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{output_folder+suptitle}.jpg")
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