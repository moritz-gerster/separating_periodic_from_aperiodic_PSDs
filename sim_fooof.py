#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 08:43:58 2021

@author: moritzgerster
"""
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


def slope_error(slopes, freq, noise_psds, freq_range):
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
    return slopes-slopes_f


def plot_all(freq, noise_psds, slopes, freq_range,
             plot_osc=False, save_path=None, save_name=None, add_title=None):
    """
    Plot original spectrum + fooof and IRASA fits and error.

    Parameters
    ----------
    freq : ndarray
        Frequency bins of psd.
    noise_psds : ndarray
        Array of noises with different 1/f slopes..
    slopes : ndarray
        1/f slopes.
    freq_range : tuple of floats
        Fitting range for both methods.
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
    fig, axes = plt.subplots(1, 3, figsize=[17, 6])

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

    err_f = slope_error(slopes, freq, noise_psds, freq_range)
    err_sum_f = np.sum(np.abs(err_f))

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
    ax.plot(slopes, slopes-slopes, "k", label="Ground Truth")
    ax.plot(slopes, err_f, "r", label="fooof")
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
    suptitle = (f"Fit Range: {freq_name}")
    if add_title:
        suptitle += ", " + add_title
    plt.suptitle(suptitle, position=[0.5, 1.02], fontsize=20)

    if save_name:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + save_name, bbox_inches="tight")
    plt.show()


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
fig_path = "../plots/"
white_ratio = 0





# %% fooof Delta Oscis


folder = "fooof_delta"
save_path = fig_path + f"{folder}/"
freq_range = [1, 45]
win_sec = 1
nperseg = int(win_sec * srate)



# Oscillation
width = .7

# Initilaiize
df = pd.DataFrame()

for delta_present in [0, 1]:

    amp = np.array([delta_present, 1, 1])

    delta = 1

    errs_f = []
    freq_osc = [delta, 10, 25]  # Hz

    scaling = np.arange(0, 12, 2)
    for scale in scaling:

        amp_scaled = amp * scale

        signals = osc_signals(samples, slopes, freq_osc, amp_scaled, width=width)
        freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
        osc_name = [f"{osc} Hz" for osc in freq_osc]
        save_name = (f"{freq_name}_osc_freqs={osc_name}_amp={amp_scaled}.pdf")
        add_title = f"osc. freqs={osc_name}, amp={amp_scaled}"

        freq, noise_psds = psds_pink(signals, srate, nperseg)
        # set lowpass filter
        lowpass = (freq < 1)
        noise_psds[:, lowpass] = 0

        err_f = slope_error(slopes, freq, noise_psds, freq_range)
        errs_f.append(np.sum(np.abs(err_f)))

        plot_all(freq, noise_psds, slopes, freq_range,
                 plot_osc=True,
                 save_path=save_path, save_name=save_name, add_title=add_title)

    data = {"freq_range": [freq_range] * len(scaling),
            "welch_window": [win_sec] * len(scaling),
            "err_f": errs_f,
            "vary": [folder] * len(scaling),
            "amp": [amp] * len(scaling),
            "freq_osc": [freq_osc] * len(scaling),
            "width": [width] * len(scaling),
            "delta_present": [delta_present] * len(scaling),
            "amp_scale": np.arange(0, 12, 2),
           }
    temp = pd.DataFrame(data)
    df = df.append(temp)


# %%
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = f"df_{folder}.pkl"
df.to_pickle(save_path + save_name)


fig, ax = plt.subplots(1, 1)

df0 = df[df.delta_present==0]
ax.plot(df0["amp_scale"], df0["err_f"], "r--",
        alpha=1, label="fooof without Delta")

df1 = df[df.delta_present==1]
ax.plot(df1["amp_scale"], df1["err_f"], "b--",
        alpha=1, label="fooof with 1Hz peak")

ax.set_xlabel("Scaling of the amplitudes")
ax.set_ylabel("Fitting error")
ax.legend()
# plt.title(f"IRASA hmin={1/hset_max:.1f}")
plt.tight_layout()
plt.savefig(fig_path + folder + ".pdf", bbox_inches="tight")
plt.show()













