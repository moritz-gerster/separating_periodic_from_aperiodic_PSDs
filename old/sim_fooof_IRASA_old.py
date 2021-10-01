"""
Simulate 1/f power spectra to compare fooof 1/f-estimation vs. IRASA.

1. Create 1/f noise for different exponents.
2. Fit fooof and IRASA. Test how well they obtain the 1/f-exponents.
4. Vary parameters such as nperseg in Welch
3. Add Gaussian white noise. Test how the noise affects the 1/f estimate
   for different frequency ranges.
5. Add one very strong sine peak. Test if 1/f is still fitted well.
   Test if fooof and IRASA detect the peak power and frequency and width
   correctly.
6. Vary the amplitude (SNR) and width of the peak.
7. Vary fitting ranges.
8. Add more peaks. Some strong, some weak, some wide, small sharp, some
   overlapping, some distinct. Which algorithm performs best?
9. Add noise to peaks
"""
import numpy as np
import scipy.signal as sig
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker
from fooof import FOOOF
from fooof import FOOOFGroup
import yasa  # IRASA
from pathlib import Path
import pandas as pd


params = {'legend.fontsize': 12.2,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'xtick.minor.size': 0,
          'xtick.major.size': 5,
          'ytick.major.size': 5,
          'lines.linewidth': 2}
plt.rcParams.update(params)


def noise_white(samples, seed=True):
    """Create White Noise of N samples."""
    if seed:
        np.random.seed(10)
    noise = np.random.normal(0, 1, size=samples)
    return noise


def osc_signals(samples, slope, freq_osc, amp, width=None, seed=True):
    """
    Generate a mixture of 1/f-aperiodic and periodic signals.

    Parameters
    ----------
    samples : int
        Number of signal samples.
    freq_osc : float or list of floats
        Peak frequencies.
    amp : float or list of floats
        Amplitudes in relation to noise.
    slope : nd.array
        1/f-slope values.
    width : None or float or list of floats, optional
        Standard deviation of Gaussian peaks. The default is None which
        corresponds to sharp delta peaks.

    Returns
    -------
    pink_noise : ndarray of size slope.size X samples
        Return signal.
    """
    if seed:
        np.random.seed(10)
    # Make fourier amplitudes
    amps = np.ones(samples//2 + 1, complex)
    freqs = np.fft.rfftfreq(samples, d=1/srate)

    # check if input is float or lists
    if isinstance(freq_osc, (int, float)):
        freq_osc = [freq_osc]  # if float, make iterable
    if isinstance(freq_osc, list):
        peaks = len(freq_osc)
    if isinstance(amp, (int, float)):
        amp = [amp] * peaks
        assert peaks == len(amp), "input lists must be of the same length"
    if isinstance(width, (int, float)):
        width = [width] * peaks
    elif isinstance(width, list):
        assert peaks == len(width), "input lists must be of the same length"
    # add Gaussian peaks to the spectrum
    if isinstance(width, list):
        for i in range(peaks):
            freq_idx = np.abs(freqs - freq_osc[i]).argmin()
            # make Gaussian peak
            if width:
                amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
                amp_dist /= np.max(amp_dist)
                amps += amp[i] * amp_dist
    # if width is none, add pure sine peaks
    elif isinstance(freq_osc, list):
        for i in range(peaks):
            freq_idx = np.abs(freqs - freq_osc[i]).argmin()
            amps[freq_idx] += amp[i]
    elif freq_osc is None:
        msg = ("what the fuck do you want? peaks or no peaks? "
               "freq_osc is None but amp is {}".format(type(amp).__name__))
        assert amp is None, msg

    amps, freqs, = amps[1:], freqs[1:]  # avoid divison by 0
    # Generate random phases
    random_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)
    if isinstance(slope, (int, float)):
        # Multiply Amp Spectrum by 1/f
        amps = amps / freqs ** (slope / 2)  # half slope needed
        amps *= np.exp(1j * random_phases)
        # Transform back to get pink noise time series
        noise = np.fft.irfft(amps)
        # normalize
        return (noise - noise.mean()) / noise.std()
    elif isinstance(slope, (np.ndarray, list)):
        pink_noises = np.zeros([len(slope), samples-2])
        for i in range(len(slope)):
            # Multiply Amp Spectrum by 1/f
            amps_i = amps / freqs ** (slope[i] / 2)  # half slope needed
            amps_i *= np.exp(1j * random_phases)
            # Transform back to get pink noise time series
            noise = np.fft.irfft(amps_i)
            # normalize
            pink_noises[i] = (noise - noise.mean()) / noise.std()
        return pink_noises


def psds_pink(noises, srate, nperseg, normalize=False):
    """
    Return freqs, and psds of noises array.

    Parameters
    ----------
    noises : ndarray
        Noise time series.
    srate : float
        Sample rate for Welch.
    nperseg : int
        Nperseg for Welch.
    normalize : boolean, optional
        Whether the psd amplitudes should all have a maximum of 1.
        The default is False.

    Returns
    -------
    tuple of ndarrays
        One frequency and N psd arrays of N noise signals.
    """
    noise_psds = []
    for i in range(noises.shape[0]):
        freq, psd = sig.welch(noises[i], fs=srate, nperseg=nperseg)
        # divide by max value to have the same offset at 2Hz
        if normalize:
            noise_psds.append(psd / psd.max())
        else:
            noise_psds.append(psd)
    return freq, np.array(noise_psds)


def slope_error(slopes, freq, noise_psds, freq_range, IRASA):
    """
    Calculate fooof and IRASA slope estimation difference to ground truth.

    Fooof needs to be calculated to obtain an array of all estimates,
    RASA need to be calculated before.

    Parameters
    ----------
    slopes : ndarray
        Ground truth.
    freq : ndarray
        Frequencies.
    noise_psds : ndarray
        PSDs.
    freq_range : tuple of floats
        Freq range used for estimation.
    IRASA : tuple
        IRASA output. Only IRASA fit params are needed.

    Returns
    -------
    ndarray
        Ground truth - fooof estimates.
    ndarray
        Ground truth - IRASA estimates.

    """
    # fg = FOOOFGroup(**fooof_params)  # Init fooof
    fg = FOOOFGroup()  # Init fooof
    fg.fit(freq, noise_psds, freq_range)
    slopes_f = fg.get_params("aperiodic", "exponent")
    _, _, _, slopes_i = IRASA
    slopes_i = slopes_i["Slope"].to_numpy()
    slopes_i = -slopes_i  # make fooof and IRASA slopes comparable
    return slopes-slopes_f, slopes-slopes_i


def plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
             plot_osc=False, save_path=None, save_name=None, add_title=None):
    """
    Plot original spectrum + fooof and IRASA fits and error.

    Parameters
    ----------
    freq : ndarray
        Frequency bins of psd.
    noise_psds : ndarray
        Array of noises with different 1/f slopes..
    IRASA : tuple
        Result of yasa.irasa function.
    slopes : ndarray
        1/f slopes.
    freq_range : tuple of floats
        Fitting range for both methods.
    white_ratio : float
        Amount of white noise. 1 corresponds to 100% (SNR 50/50).
    plot_osc : boolean, optional
        Whether to plot oscillatory peaks for IRASA. The default is False.
    save_path : str, optional
        Save path. The default is None.
    save_name : str, optional
        Save name. The default is None.
    add_title : str, optional
        Title to add for description. The default is None.

    Returns
    -------
    Fig.
    """
    fig, axes = plt.subplots(1, 4, figsize=[17, 6])

    ax = axes[0]
    mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
    for i in range(slopes.size):
        ax.loglog(freq[mask], noise_psds[i, mask],
                  label=f"1/f={slopes[i]:.2f}")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ylim = ax.get_ylim()
    xticks = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 4)
    ax.set_xticks(xticks)
    xticklabels = [np.round(xtick) for xtick in xticks]
    ax.set_xticklabels(xticklabels)
    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()
    ax.set_ylabel("Power")
    ax.set_xlabel('Frequency')
    ax.set_title("1/f Noise: Ground truth")
    ax.legend(title="Ground truth", loc=3, ncol=2,
              bbox_transform=fig.transFigure,
              bbox_to_anchor=[0.12, -.285])

    err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
    err_sum_f, err_sum_i = np.sum(np.abs(err_f)), np.sum(np.abs(err_i))

    ax = axes[1]
    labels = []
    for i, noise_psd in enumerate(noise_psds):
        print(f"...fitting fooof {i+1} of {len(slopes)}")
        # fm = FOOOF(**fooof_params)  # Init fooof
        fm = FOOOF()  # Init fooof
        try:
            fm.fit(freq, noise_psd, freq_range)
            fm.plot(plt_log=True, ax=ax)
            exponent = fm.get_params('aperiodic_params', 'exponent')
            labels.append(f" 1/f={exponent:.2f}")
        except:
            # offset = fm.get_params('aperiodic_params', 'offset')
            labels.append(" 1/f=failed")
    handles, _ = ax.get_legend_handles_labels()
    handles = handles[2::3] + [handles[-2]]
    labels = labels + ["osc"]
    ax.legend(handles, labels, title="fooof", loc=3, ncol=2,
              bbox_transform=fig.transFigure,
              bbox_to_anchor=[0.322, -.285])
    freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
    ax.set_title(f"Fooof error: {err_sum_f:.2f}")
    ax.set_ylabel("")
    ax.set_xlabel('Frequency', fontsize='x-large')
    ax.set_yticks(np.log10(yticks))
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(np.log10(xticks))
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(np.log10(ylim))
    ax.grid(False)


    ax = axes[2]
    freq_i, aperiodic, osc, params = IRASA
    # normalize
    # if normalize:
    #    aperiodic /= aperiodic.max(1)[:, np.newaxis]
    for i in range(slopes.size):
        ax.loglog(freq_i, aperiodic[i], "b", linestyle="--", lw=2,
                  label="aperiodic")
        if plot_osc:
            ax.loglog(freq_i, osc[i] + aperiodic[i], "r", lw=2,
                      label="osc", alpha=.5)
    handles, _ = ax.get_legend_handles_labels()
    labels = [f"1/f={-params['Slope'].iloc[i]:.2f}"
                    # f"     Offset={params['Intercept'].iloc[i]:.2f}"
                    for i in range(slopes.size)]
    if plot_osc:
        labels = labels + ["osc"]
        handles = handles[::2] + [handles[-1]]
    else:
        handles = handles[::2]
    ax.legend(handles, labels, title="IRASA", loc=3, ncol=2,
              bbox_transform=fig.transFigure,
              bbox_to_anchor=[0.525, -.285])
    # ax.set_ylim([ymin, ymax])
    # ax.set_xlim([1, 600])
    ax.set_title(f"IRASA error: {err_sum_i:.2f}")
    ax.set_xlabel('Frequency')
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    # ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax = axes[3]
    ax.plot(slopes, slopes-slopes, "k", label="Ground Truth")
    ax.plot(slopes, err_f, "r", label="fooof")
    ax.plot(slopes, err_i, "b", label="IRASA")
    ax.legend()
    ax.set_xticks(slopes[::2])
    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()
    ax.secondary_yaxis('right')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel("1/f")
    # ax.set_ylabel("Fitting error")
    ax.set_title("Ground truth - fitting value")
    suptitle = (f"Fit Range: {freq_name}, White noise: {white_ratio*100}%, "
                f"Welch window: {win_sec}s")
    if add_title:
        suptitle += ", " + add_title
    plt.suptitle(suptitle, position=[0.5, 1.02], fontsize=20)

    if save_name:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + save_name, bbox_inches="tight")
    plt.show()


# =============================================================================
# def plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
#              save_path=None, save_name=None, add_title=None):
#     """
#     Plot original spectrum + fooof and IRASA fits and error.
# 
#     Parameters
#     ----------
#     freq : ndarray
#         Frequency bins of psd.
#     noise_psds : ndarray
#         Array of noises with different 1/f slopes..
#     IRASA : tuple
#         Result of yasa.irasa function.
#     slopes : ndarray
#         1/f slopes.
#     freq_range : tuple of floats
#         Fitting range for both methods.
#     white_ratio : float
#         Amount of white noise. 1 corresponds to 100% (SNR 50/50).
#     save_path : str, optional
#         Save path. The default is None.
#     save_name : str, optional
#         Save name. The default is None.
#     add_title : str, optional
#         Title to add for description. The default is None.
# 
#     Returns
#     -------
#     Fig.
#     """
#     fig, axes = plt.subplots(1, 4, figsize=[16, 6])
# 
#     ax = axes[0]
#     mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
#     for i in range(slopes.size):
#         ax.loglog(freq[mask], noise_psds[i, mask],
#                   label=f"1/f={slopes[i]:.2f}")
#     ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
#     ylim = ax.get_ylim()
#     xticks = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 4)
#     ax.set_xticks(xticks)
#     xticklabels = [int(np.round(xtick)) for xtick in xticks]
#     ax.set_xticklabels(xticklabels)
#     yticks = ax.get_yticks()
#     yticklabels = ax.get_yticklabels()
#     ax.set_ylabel("Power")
#     ax.set_xlabel('Frequency')
#     ax.set_title("1/f Noise: Ground truth")
#     ax.legend(title="Ground truth", loc=3, ncol=2,
#               bbox_transform=fig.transFigure,
#               bbox_to_anchor=[0.12, -.285])
# 
#     err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
#     err_sum_f, err_sum_i = np.sum(np.abs(err_f)), np.sum(np.abs(err_i))
# 
#     ax = axes[1]
#     labels = []
#     for i, noise_psd in enumerate(noise_psds):
#         print(f"...fitting fooof {i+1} of {len(slopes)}")
#         # fm = FOOOF(**fooof_params)  # Init fooof
#         fm = FOOOF()  # Init fooof
#         try:
#             fm.fit(freq, noise_psd, freq_range)
#             fm.plot(plt_log=True, ax=ax)
#             exponent = fm.get_params('aperiodic_params', 'exponent')
#             labels.append(f" 1/f={exponent:.2f}")
#         except:
#             # offset = fm.get_params('aperiodic_params', 'offset')
#             labels.append(" 1/f=failed")
#     handles, _ = ax.get_legend_handles_labels()
#     handles = handles[2::3] + [handles[-2]]
#     labels = labels + ["osc"]
#     ax.legend(handles, labels, title="fooof", loc=3, ncol=2,
#               bbox_transform=fig.transFigure,
#               bbox_to_anchor=[0.322, -.285])
#     freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
#     ax.set_title(f"Fooof error: {err_sum_f:.2f}")
#     ax.set_ylabel("")
#     ax.set_xlabel('Frequency', fontsize='x-large')
#     ax.set_yticks(np.log10(yticks))
#     ax.set_yticklabels(yticklabels)
#     ax.set_xticks(np.log10(xticks))
#     ax.set_xticklabels(xticklabels)
#     ax.set_ylim(np.log10(ylim))
#     ax.grid(False)
# 
#     if IRASA:
#         ax = axes[2]
#         freq_i, aperiodic, osc, params = IRASA
#         # normalize
#         # if normalize:
#         #    aperiodic /= aperiodic.max(1)[:, np.newaxis]
#         for i in range(slopes.size):
#             ax.loglog(freq_i, aperiodic[i], "b", linestyle="--", lw=2,
#                       label="aperiodic")
#             # ax.loglog(freq_i, osc[i], "lightgrey", lw=2, label="oscillatory")
#             ax.loglog(freq_i, osc[i] + aperiodic[i], "r", lw=2, label="osc",
#                       alpha=.5)
#         handles, labels = ax.get_legend_handles_labels()
#         slope_labels = [f"1/f={-params['Slope'].iloc[i]:.2f}"
#                         # f"     Offset={params['Intercept'].iloc[i]:.2f}"
#                         for i in range(slopes.size)]
#         # handles = handles[1:3] + handles[::3]
#         # labels = labels[1:3] + slope_labels
#         labels = slope_labels + [labels[-1]]
#         handles = handles[::2] + [handles[-1]]
#         ax.legend(handles, labels, title="IRASA", loc=3, ncol=2,
#                   bbox_transform=fig.transFigure,
#                   bbox_to_anchor=[0.525, -.285])
#         # ax.set_ylim([ymin, ymax])
#         # ax.set_xlim([1, 600])
#         ax.set_title(f"IRASA error: {err_sum_i:.2f}")
#         ax.set_xlabel('Frequency')
#         ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xticklabels)
#         # ax.set_yticks(yticks)
#         ax.set_yticklabels(yticklabels)
# 
#     ax = axes[3]
#     ax.plot(slopes, slopes-slopes, "k", label="Ground Truth")
#     ax.plot(slopes, err_f, "r", label="fooof")
#     ax.plot(slopes, err_i, "b", label="IRASA")
#     ax.legend()
#     ax.set_xticks(slopes[::2])
#     yticks = ax.get_yticks()
#     yticklabels = ax.get_yticklabels()
#     ax.secondary_yaxis('right')
#     ax.set_yticks([])
#     ax.set_yticklabels([])
#     ax.set_xlabel("1/f")
#     # ax.set_ylabel("Fitting error")
#     ax.set_title("Ground truth - fitting value")
# 
#     suptitle = (f"Fit Range: {freq_name}, White noise: {white_ratio*100}%, "
#                 f"Welch window: {win_sec}s")
#     if add_title:
#         suptitle += ", " + add_title
#     plt.suptitle(suptitle, position=[0.5, 1.02], fontsize=20)
#     # plt.tight_layout()
#     plt.subplots_adjust(wspace=0.05)
#     if save_name:
#         Path(save_path).mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path + save_name, bbox_inches="tight")
#     plt.show()
# =============================================================================



# %%PARAMETERS


# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slopes = np.arange(0, 4.5, .5)

# WELCH
win_sec = 4
nperseg = int(win_sec * srate)

# Fit Params
# fooof_params = {"peak_width_limits": (4, 12)}
irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                "kwargs_welch": {'average': 'mean'}}

# Path
fig_path = "../plots_old/"
white_ratio = 0



# %% White noise variation






folder = "white_noise"
save_path = fig_path + f"{folder}/"
freq_range = [2, 100]

white_ratios = [0, 0.001, 0.05, 1]

# Make noise
noises = osc_signals(samples, slopes, None, None)
w_noise = noise_white([slopes.size, samples-2])

# Initilaiize
errs_f = []
errs_i = []
for white_ratio in white_ratios:
    noise_mix = noises + white_ratio * w_noise
    freq, noise_psds = psds_pink(noise_mix, srate, nperseg)

    save_name = f"noise={white_ratio}.pdf"
    IRASA = yasa.irasa(data=noise_mix, band=freq_range, **irasa_params)

    err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
             save_path=save_path, save_name=save_name)

data = {"freq_ranges": [freq_range] * len(white_ratios),
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(white_ratios),
        "white_ratio": white_ratios}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_noise_vary.pkl"
df.to_pickle(save_path + save_name)


fig, ax = plt.subplots(1, 1)
ax.plot(df["white_ratio"], df["err_f"], "r--", label="fooof")
ax.plot(df["white_ratio"], df["err_i"], "b:", label="IRASA")
ax.set_xlabel("White Noise ratio")
ax.set_ylabel("Fitting error")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
plt.title("Realistic fitting ranges")
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()








# %% Welch variation



folder = "welch_window"
save_path = fig_path + f"{folder}/"
freq_range = [1, 100]
welch_windows = [0.25, 0.5, 1, 2, 4, 8]

# Make noise
noises = osc_signals(samples, slopes, None, None)
white_ratio = 0

# Initilaiize
errs_f = []
errs_i = []
for win_sec in welch_windows:
    nperseg = int(win_sec * srate)
    freq, noise_psds = psds_pink(noises, srate, nperseg)
    irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                    "kwargs_welch": {'average': 'mean'}}

    save_name = f"welch_window={win_sec}.pdf"
    IRASA = yasa.irasa(data=noises, band=freq_range, **irasa_params)

    err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
             save_path=save_path, save_name=save_name)

data = {"freq_ranges": [freq_range] * len(welch_windows),
        "welch_window": welch_windows,
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(welch_windows),
        "noise_gauss": [white_ratio] * len(welch_windows)}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_welch_window_vary.pkl"
df.to_pickle(save_path + save_name)


fig, ax = plt.subplots(1, 1)
ax.plot(df["welch_window"], df["err_f"], "r--",
        alpha=1, label="fooof")
ax.plot(df["welch_window"], df["err_i"], "b:",
        alpha=1, label="IRASA")
ax.set_xlabel("Welch Window [s]")
ax.set_ylabel("Fitting error")
ax.legend()
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()











# %% IRASA NOT RELIABLE FOR HIGH FREQUENCIES







folder = "IRASA_high_freq"
save_path = fig_path + f"{folder}/"
freq_ranges = [[10, 110], [210, 310], [310, 410], [360, 460],
               [410, 510], [460, 560], [510, 610]]

# Make noise
noises = osc_signals(samples, slopes, None, None)
white_ratio = 0

# Initilaiize
errs_f = []
errs_i = []
for freq_range in freq_ranges:
    freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
    save_name = f"{freq_name}.pdf"
    freq, noise_psds = psds_pink(noises, srate, nperseg)
    IRASA = yasa.irasa(data=noises, band=freq_range, **irasa_params)

    err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
             plot_osc=True,
             save_path=save_path, save_name=save_name)


data = {"freq_ranges": freq_ranges,
        "welch_window": [win_sec] * len(freq_ranges),
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(freq_ranges),
        "noise_gauss": [white_ratio] * len(freq_ranges)}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_high_freq_vary.pkl"
df.to_pickle(save_path + save_name)


fig, ax = plt.subplots(1, 1)
for i in range(df.shape[0]):
    ax.plot(df["freq_ranges"][i], [df["err_f"][i], df["err_f"][i]], "r--",
            alpha=1, label="fooof")
    ax.plot(df["freq_ranges"][i], [df["err_i"][i], df["err_i"][i]], "b:",
            alpha=1, label="IRASA")
ax.set_xlabel("Frequency Range [Hz]")
ax.set_ylabel("Fitting error")
ymin, ymax = ax.get_ylim()
ax.set_ylim([ymin, ymax])
ax.vlines((srate/1.9)/2, ymin, ymax, color="k", label="Resample Nyquist Freq.")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2] + [handles[-1]], labels[:2] + [labels[-1]])
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()





# %% Add sine peaks for given parameters









folder = "pure_sine_peaks"
save_path = fig_path + f"{folder}/"

freq_range = [30, 50]

# Generate signal
freq = 40 # Hz
amp = 100
signals = osc_signals(samples, slopes, freq, amp)

freq, noise_psds = psds_pink(signals, srate, nperseg)
freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
save_name = f"{freq_name}.pdf"
#IRASA = yasa.irasa(data=signals, band=freq_range, **irasa_params)

plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
         plot_osc=True, save_path=save_path, save_name=save_name)












# %% IRASA neees larger up/down sampling for more broad widths




folder = "oscillation_widths"
save_path = fig_path + f"{folder}/"
freq_range = [5, 40]
white_ratio = 0
win_sec = 4


# Oscillation
freq_osc = [10, 25]  # Hz
amp = [2, 10]

# Initilaiize
df = pd.DataFrame()
hset_maxis = [1.9, srate / 4 / freq_range[1]]
for hset_max in hset_maxis:
    irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                    "kwargs_welch": {'average': 'mean'},
                    "hset": np.linspace(1.1, hset_max, num=18)}
    beta_widths = [0.1, 2, 4, 6, 8]
    errs_f = []
    errs_i = []
    for beta_width in beta_widths:
        width = [0.5, beta_width]
        signals = osc_signals(samples, slopes, freq_osc, amp, width=width)
        freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
        save_name = (f"{freq_name}_beta_width={beta_width}_"
                     f"hset_max={hset_max:.2f}.pdf")
        add_title = f"beta width={beta_width} SDs, hmax={hset_max:.2f}"
        freq, noise_psds = psds_pink(signals, srate, nperseg)
        IRASA = yasa.irasa(data=signals, band=freq_range, **irasa_params)
        err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
        errs_f.append(np.sum(np.abs(err_f)))
        errs_i.append(np.sum(np.abs(err_i)))

        # plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
          #       save_path=save_path, save_name=save_name, add_title=add_title)
        plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
                 plot_osc=True,
                 save_path=save_path, save_name=save_name, add_title=add_title)

    data = {"freq_ranges": [freq_range] * len(beta_widths),
            "welch_window": [win_sec] * len(beta_widths),
            "err_f": errs_f, "err_i": errs_i,
            "vary": [folder] * len(beta_widths),
            "amp": [amp] * len(beta_widths),
            "freq_osc": [freq_osc] * len(beta_widths),
            "width": beta_widths,
            "hset_max": [hset_max] * len(beta_widths),
            "noise_gauss": [white_ratio] * len(beta_widths)}
    temp = pd.DataFrame(data)
    df = df.append(temp)

save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_beta_width_hmax_vary.pkl"
df.to_pickle(save_path + save_name)

df1 = df[df.hset_max == 1.9]
df2 = df[df.hset_max == 15]

fig, ax = plt.subplots(1, 1)
ax.plot(df1["width"], df1["err_f"], "r--",
        alpha=1, label="fooof")
ax.plot(df1["width"], df1["err_i"], "b:",
        alpha=1, label="IRASA hmax=1.9")
ax.plot(df2["width"], df2["err_i"], "b-",
        alpha=1, label="IRASA hmax=15")
ax.set_xlabel("Gauss width SDs")
ax.set_ylabel("Fitting error")
ax.legend()
# plt.title(f"IRASA hmin={1/hset_max:.1f}")
plt.tight_layout()
plt.savefig(fig_path + folder + "/hset_max_summary.pdf", bbox_inches="tight")
plt.show()











# %% peak_extends_fitting_range




folder = "peak_extends_fitting_range"
save_path = fig_path + f"{folder}/"
freq_ranges = [[1, 50], [2, 50], [3, 50], [4, 50], [5, 50], [6, 50], [7, 50],
               [10, 50], [12, 50], [20, 50]]

freq_range = [2, 50]
hset_max = srate / 4 / freq_range[1]
irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                    "kwargs_welch": {'average': 'mean'},
                    "hset": np.linspace(1.1, hset_max, num=18)}

# Oscillation
# freq_osc = [3, 6, 10, 18, 20]  # Hz
# amp = [4, 4, 4, 5, 10]
# width = [.5, .3, .5, 2, 3]
freq_osc = [3, 6, 10, 20]  # Hz
amp = [4, 4, 4, 7]
width = [.5, .3, .5, 3]

# Initilaiize
errs_f = []
errs_i = []
for freq_range in freq_ranges:
    freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
    save_name = f"{freq_name}.pdf"
    add_title= f"Peaks at 3, 6, 10, and 20Hz"
    signals = osc_signals(samples, slopes, freq_osc, amp, width=width)
    freq, noise_psds = psds_pink(signals, srate, nperseg)
    IRASA = yasa.irasa(data=signals, band=freq_range, **irasa_params)

    err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    # plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
      #       save_path=save_path, save_name=save_name, add_title=add_title)
    plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
             plot_osc=True,
             save_path=save_path, save_name=save_name, add_title=add_title)

data = {"freq_ranges": freq_ranges,
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(freq_ranges),
        "win_sec": [win_sec] * len(freq_ranges),
        "resampling_max": [hset_max] * len(freq_ranges),
        "amp": [amp] * len(freq_ranges),
        "widths": [width] * len(freq_ranges),
        "freq_oscs": [freq_osc] * len(freq_ranges),
        "noise_gauss": [white_ratio] * len(freq_ranges)}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_fit_range_low.pkl"
df.to_pickle(save_path + save_name)

fig, ax = plt.subplots(1, 1)

df["f_low"] = np.array(freq_ranges)[:, 0]

ax.plot(df["f_low"], df["err_f"], "r--",
        alpha=1, label="fooof")
ax.plot(df["f_low"], df["err_i"], "b:",
        alpha=1, label="IRASA")
ax.set_xticks(freq_osc)
ax.set_xlim([0, 13])
ax.set_ylim([0, 13])
ax.set_xlabel("Lower Frequency Fitting Border [Hz]")
ax.set_ylabel("Fitting error")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
plt.title(add_title)
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()








# %% SNR




folder = "SNR"
save_path = fig_path + f"{folder}/"


freq_range = [5, 45]
hset_max = srate / 4 / freq_range[1]
irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                    "kwargs_welch": {'average': 'mean'},
                    "hset": np.linspace(1.1, hset_max, num=18)}

# Oscillation
freq_osc = [10, 18, 20]  # Hz
amp = np.array([2, 5, 10])
width = [.5, 2, 4]

# Initilaiize
errs_f = []
errs_i = []
amp_mods = [0.01, .1, .5]
for amp_mod in amp_mods:
    freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
    save_name = f"{freq_name}_amp-mod={amp_mod}.pdf"
    signals = osc_signals(samples, slopes, freq_osc, amp*amp_mod, width=width)
    freq, noise_psds = psds_pink(signals, srate, nperseg)
    IRASA = yasa.irasa(data=signals, band=freq_range, **irasa_params)

    err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    # plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
      #       save_path=save_path, save_name=save_name)
    plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
             plot_osc=True,
             save_path=save_path, save_name=save_name)

data = {"freq_ranges": [freq_range] * len(amp_mods),
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(amp_mods),
        "win_sec": [win_sec] * len(amp_mods),
        "resampling_max": [hset_max] * len(amp_mods),
        "amp": [amp] * len(amp_mods),
        "amp_mods": amp_mods,
        "widths": [width] * len(amp_mods),
        "freq_oscs": [freq_osc] * len(amp_mods),
        "noise_gauss": [white_ratio] * len(amp_mods)}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_SNR.pkl"
df.to_pickle(save_path + save_name)

fig, ax = plt.subplots(1, 1)

ax.plot(df["amp_mods"], df["err_f"], "r--",
        alpha=1, label="fooof")
ax.plot(df["amp_mods"], df["err_i"], "b:",
        alpha=1, label="IRASA")
ax.set_xlabel("Amplitude modulation a.u.")
ax.set_ylabel("Fitting error")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
plt.title("Oscillatory peaks at 10, 18, and 20Hz")
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()








# %% realistic+noise


folder = "realistic+noise"
save_path = fig_path + f"{folder}/"
freq_range = [5, 45]

# Oscillation
freq_osc = [10, 18, 20]  # Hz
amp = [2, 5, 10]
width = [.5, 2, 4]
white_ratios = [0.001, .01, .05, .1, .2]

# Initilaiize
errs_f = []
errs_i = []
w_noise = noise_white([slopes.size, samples-2])

for white_ratio in white_ratios:
    freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
    save_name = f"{freq_name}_amp-mod={amp_mod}.pdf"
    signals = osc_signals(samples, slopes, freq_osc, amp, width=width)
    signals += white_ratio * w_noise
    freq, noise_psds = psds_pink(signals, srate, nperseg)
    IRASA = yasa.irasa(data=signals, band=freq_range, **irasa_params)

    err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    # plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
      #       save_path=save_path, save_name=save_name)
    plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
             plot_osc=True,
             save_path=save_path, save_name=save_name)

data = {"freq_ranges": [freq_range] * len(white_ratios),
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(white_ratios),
        "win_sec": win_sec * len(white_ratios),
        "resampling_max": [hset_max] * len(white_ratios),
        "amp": [amp] * len(white_ratios),
        "widths": [width] * len(white_ratios),
        "freq_oscs": [freq_osc] * len(white_ratios),
        "noise_gauss": white_ratios}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_noise.pkl"
df.to_pickle(save_path + save_name)


fig, ax = plt.subplots(1, 1)
ax.plot(df["noise_gauss"], df["err_f"], "r--",
        alpha=1, label="fooof")
ax.plot(df["noise_gauss"], df["err_i"], "b:",
        alpha=1, label="IRASA")
xticks = ax.get_xticks()
ax.set_xticklabels([int(xtick*100) for xtick in xticks])
ax.set_xlabel("White Noise [%]")
ax.set_ylabel("Fitting error")
ax.legend()
# plt.title("Realistic fitting ranges")
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()













# %% IRASA fucks up when highpass filtering


folder = "highpass"
save_path = fig_path + f"{folder}/"
freq_range = [5, 95]
white_ratio = 0
win_sec = 4


# Oscillation
freq_osc = [10, 25]  # Hz
amp = [2, 10]


hset_max = srate / 4 / freq_range[1]
hset_max = 9.9

irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                "kwargs_welch": {'average': 'mean'},
                "hset": np.linspace(1.1, hset_max, num=5)}
# beta_widths = [0.1, 2, 4]
beta_width = 2
width = [0.5, beta_width]
signals = osc_signals(samples, slopes, freq_osc, amp, width=width)

# highpass filter
sos = sig.butter(10, 1, btype="hp", fs=srate, output='sos')
signals = sig.sosfilt(sos, signals)
freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
save_name = (f"{freq_name}_beta_width={beta_width}_"
             f"hset_max={hset_max:.2f}.pdf")
add_title = f"beta width={beta_width} SDs, hmax={hset_max:.2f}"
freq, noise_psds = psds_pink(signals, srate, nperseg)
IRASA = yasa.irasa(data=signals, band=freq_range, **irasa_params)

plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
         plot_osc=True,
         save_path=save_path, save_name=save_name, add_title=add_title)

# %% IRASA fucks up when noise floor


folder = "noise_floor"
save_path = fig_path + f"{folder}/"
freq_range = [15, 200]
white_ratio = .6
win_sec = 1
nperseg = int(win_sec * srate)

slopes = np.array([1, 1.6])
# Oscillation
freq_osc = 50  # Hz
amp = 1
w_noise = noise_white([slopes.size, samples-2])


# hset_max = srate / 4 / freq_range[1]
hset_max = 9.9

irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                "kwargs_welch": {'average': 'mean'},
                "hset": np.linspace(1.1, hset_max, num=5)}

width = 6
signals = osc_signals(samples, slopes, freq_osc, amp, width=width)
signals = signals + white_ratio * w_noise

# highpass filter
sos = sig.butter(10, 1, btype="hp", fs=srate, output='sos')
signals = sig.sosfilt(sos, signals)
freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
save_name = (f"{freq_name}_beta_width={width}_"
             f"hset_max={hset_max:.2f}.pdf")
add_title = f"beta width={width} SDs, hmax={hset_max:.2f}"
freq, noise_psds = psds_pink(signals, srate, nperseg)
IRASA = yasa.irasa(data=signals, band=freq_range, **irasa_params)

plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
         plot_osc=True,
         save_path=save_path, save_name=save_name, add_title=add_title)



# %% IRASA fine when using intermediate ranges?


folder = "IRASA_works"
save_path = fig_path + f"{folder}/"
freq_range = [15, 30]
white_ratio = 0
win_sec = 4
nperseg = int(win_sec * srate)


slopes = np.arange(0, 4.5, .5)

# Oscillation
freq_osc = [14, 18, 27]
amp = [1, 1, 1]
width = [4, 2, 4]

slopes = np.array([.8, 1.2])


# hset_max = srate / 4 / freq_range[1]
hset_max = 9.9

irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                "kwargs_welch": {'average': 'mean'},
                "hset": np.linspace(1.1, hset_max, num=15)}

signals = osc_signals(samples, slopes, freq_osc, amp, width=width)


# highpass filter
sos = sig.butter(10, 1, btype="hp", fs=srate, output='sos')
signals = sig.sosfilt(sos, signals)
freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
save_name = (f"{freq_name}_beta_width={width}_"
             f"hset_max={hset_max:.2f}.pdf")
add_title = f"beta width={width} SDs, hmax={hset_max:.2f}"
freq, noise_psds = psds_pink(signals, srate, nperseg)
IRASA = yasa.irasa(data=signals, band=freq_range, **irasa_params)

plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
         plot_osc=True,
         save_path=save_path, save_name=save_name, add_title=add_title)

# %%
hset_max = 2
freq, noise_psds = psds_pink(signals, srate, nperseg)
freq2, noise_psds2 = psds_pink(signals, srate/hset_max, nperseg)
freq3, noise_psds3 = psds_pink(signals, srate*hset_max, nperseg)
plt.plot(freq[:150], noise_psds[3][:150]/noise_psds[3][:150].max(),
         freq2[:150], noise_psds2[3][:150]/noise_psds2[3][:150].max(),
         freq3[:150], noise_psds3[3][:150]/noise_psds3[3][:150].max())
plt.xticks(np.arange(0, 80, 10))

# %% Old


# =============================================================================
# # %%
# # Add sine peaks for given parameters
# # Step 2: vary Gauss amp
# 
# # oscillatory parameters:
# folder = "oscillation_Gauss_amps"
# freq_range = [30, 50]
# # freq_ranges = [[2, 40], [30, 50]]
# 
# # Oscillation
# freq_osc = 40 # Hz
# amp = 100
# 
# # Initilaiize
# df = pd.DataFrame()
# errs_f = []
# errs_i = []
# # widths = [10, 100, 1000, 10000]
# width = 1
# amps = [10, 100, 1000, 10000]
# for amp in amps:
#     signals = np.array([osc_signals(samples, freq_osc, amp, slope, width=width)
#                         for slope in slopes])
#     freq, noise_psds = psds_pink(signals, slopes, srate, nperseg)
#     freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
#     para_str = f"amps_gauss_{freq_name}.pdf"
#     IRASA = yasa.irasa(data=signals, band=freq_range, **irasa_params)
#     err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
#     errs_f.append(np.sum(np.abs(err_f)))
#     errs_i.append(np.sum(np.abs(err_i)))
# 
#     plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
#              fooof=True, para_str=para_str)
# 
# data = {"freq_ranges": [freq_range] * len(amps),
#         "err_f": errs_f, "err_i": errs_i,
#         "vary": [folder] * len(amps),
#         "win_sec": win_sec * len(amps),
#         "amp": amps,
#         "width": width * len(amps),
#         "noise_gauss": [white_ratio] * len(amps)}
# df = pd.DataFrame(data)
# save = data_path + folder + "/"
# Path(save).mkdir(parents=True, exist_ok=True)
# name = f"sim_osc_white_noise={white_ratio}_width_gauss_vary.pkl"
# df.to_pickle(save + name)
# 
# 
# folder = "oscillation_Gauss_amps"
# name = f"sim_osc_white_noise={white_ratio}_width_gauss_vary.pkl"
# load = data_path + folder + "/" + name
# df = pd.read_pickle(load)
# 
# fig, ax = plt.subplots(1, 1)
# ax.plot(df["amp"], df["err_f"], "r--",
#         alpha=1, label="fooof")
# ax.plot(df["amp"], df["err_i"], "b:",
#         alpha=1, label="IRASA")
# ax.set_xlabel("Amp Osc/Amp Noise")
# ax.set_ylabel("Fitting error")
# ax.legend()
# # plt.title("Realistic fitting ranges")
# plt.tight_layout()
# plt.savefig(fig_path + folder + ".pdf", bbox_inches="tight")
# plt.show()
# =============================================================================


# =============================================================================
# # %%
# # Add sine peaks for given parameters
# # Step 2: vary amp -> Amplitudes/SNR irrelevant
# 
# # oscillatory parameters:
# folder = "oscillations_amp"
# freq_range = [30, 50]
# # freq_ranges = [[2, 40], [30, 50]]
# 
# # Oscillation
# freq_osc = 40 # Hz
# 
# # Initilaiize
# df = pd.DataFrame()
# errs_f = []
# errs_i = []
# amps = [10, 100, 1000, 10000]
# for amp in amps:
#     signals = np.array([osc_signals(samples, freq_osc, amp, slope) for slope in slopes])
#     freq, noise_psds = psds_pink(signals, slopes, srate, nperseg)
#     freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
#     para_str = f"amps_{freq_name}_noise={white_ratio}_welch_window={win_sec}_osci.pdf"
#     IRASA = yasa.irasa(data=signals, band=freq_range, **irasa_params)
#     err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
#     errs_f.append(np.sum(np.abs(err_f)))
#     errs_i.append(np.sum(np.abs(err_i)))
# 
#     plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
#              fooof=True, para_str=para_str)
# 
# data = {"freq_ranges": [freq_range] * len(amps),
#         "err_f": errs_f, "err_i": errs_i,
#         "vary": [folder] * len(amps),
#         "win_sec": win_sec * len(amps),
#         "amp": amps,
#         "noise_gauss": [white_ratio] * len(amps)}
# df = pd.DataFrame(data)
# save = data_path + folder + "/"
# Path(save).mkdir(parents=True, exist_ok=True)
# name = f"sim_osc_white_noise={white_ratio}_amp_vary.pkl"
# df.to_pickle(save + name)
# 
# 
# folder = "oscillations_amp"
# name = f"sim_osc_white_noise={white_ratio}_amp_vary.pkl"
# load = data_path + folder + "/" + name
# df = pd.read_pickle(load)
# 
# fig, ax = plt.subplots(1, 1)
# ax.plot(df["amp"], df["err_f"], "r--",
#         alpha=1, label="fooof")
# ax.plot(df["amp"], df["err_i"], "b:",
#         alpha=1, label="IRASA")
# ax.set_xlabel("Amp Osc/Amp Noise")
# ax.set_ylabel("Fitting error")
# ax.legend()
# # plt.title("Realistic fitting ranges")
# plt.tight_layout()
# plt.savefig(fig_path + folder + ".pdf", bbox_inches="tight")
# plt.show()
# =============================================================================

# =============================================================================
# # BOTH METHODS RELIABLE FOR SMALL RANGES
# folder = "short_fit_ranges"
# save_path = fig_path + f"{folder}/"
# # reset paramters
# win_sec = 4
# nperseg = int(win_sec * srate)
# irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
#                 "kwargs_welch": {'average': 'mean'}}
# freq_ranges = [[50, 100], [50, 60], [50, 55]]
# 
# # Make noise
# noises = osc_signals(samples, slopes, None, None)
# white_ratio = 0
# 
# 
# # Initilaiize
# df = pd.DataFrame()
# errs_f = []
# errs_i = []
# for freq_range in freq_ranges:
#     freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
#     save_name = f"{freq_name}.pdf"
#     IRASA = yasa.irasa(data=noises, band=freq_range, **irasa_params)
# 
#     err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA)
#     errs_f.append(np.sum(np.abs(err_f)))
#     errs_i.append(np.sum(np.abs(err_i)))
# 
#     plot_all(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
#              save_path=save_path, save_name=save_name)
# 
# data = {"freq_ranges": freq_ranges,
#         "welch_window": win_sec * len(freq_ranges),
#         "err_f": errs_f, "err_i": errs_i,
#         "vary": [folder] * len(freq_ranges),
#         "noise_gauss": [white_ratio] * len(freq_ranges)}
# df = pd.DataFrame(data)
# save_path = fig_path + folder + "/"
# Path(save_path).mkdir(parents=True, exist_ok=True)
# save_name = "df_short_freq_vary.pkl"
# df.to_pickle(save_path + save_name)
# 
# fig, ax = plt.subplots(1, 1)
# for i in range(df.shape[0]):
#     ax.plot(df["freq_ranges"][i], [df["err_f"][i], df["err_f"][i]], "r--",
#             alpha=1, label="fooof")
#     ax.plot(df["freq_ranges"][i], [df["err_i"][i], df["err_i"][i]], "b:",
#             alpha=1, label="IRASA")
# ax.set_xlabel("Frequency Range [Hz]")
# ax.set_ylabel("Fitting error")
# ax.set_xticks([50, 55, 60, 100])
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:2], labels[:2])
# plt.title("Precision as function of fitting length")
# plt.tight_layout()
# plt.savefig(fig_path + folder + ".pdf", bbox_inches="tight")
# plt.show()
# =============================================================================