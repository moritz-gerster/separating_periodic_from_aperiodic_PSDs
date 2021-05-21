#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:24:46 2021

@author: moritzgerster
"""
"""Fooof needs clearly separable (and ideally Gaussian) peaks."""
import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig
from scipy.signal import sawtooth
import matplotlib.pyplot as plt
import matplotlib as mpl
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic


# =============================================================================
# def osc_signals(slope, periodic_params=None, nlv=None, highpass=True,
#                 srate=2400, duration=180, seed=1):
#     """
#     Generate colored noise with optionally added oscillations.
# 
#     Parameters
#     ----------
#     slope : float, optional
#         Aperiodic 1/f exponent. The default is 1.
#     periodic_params : list of tuples, optional
#         Oscillations parameters as list of tuples in form
#         [(frequency, amplitude, width), (frequency, amplitude, width)] for
#         two oscillations.
#         The default is None.
#     nlv : float, optional
#         Level of white noise. The default is None.
#     highpass : int, optional
#         The order of the butterworth highpass filter. The default is 4. If
#         None, no filter will be applied.
#     srate : float, optional
#         Sample rate of the signal. The default is 2400.
#     duration : float, optional
#         Duration of the signal in seconds. The default is 180.
#     seed : int, optional
#         Seed for reproducability. The default is 1.
# 
#     Returns
#     -------
#     noise : ndarray
#         Colored noise without oscillations.
#     noise_osc : ndarray
#         Colored noise with oscillations.
#     """
#     if seed:
#         np.random.seed(seed)
#     # Initialize
#     n_samples = int(duration * srate)
#     amps = np.ones(n_samples//2 + 1, complex)
#     freqs = rfftfreq(n_samples, d=1/srate)
#     freqs = freqs[1:]  # avoid divison by 0
#     # freqs[0] = 1  # avoid divison by 0
# 
#     # Create random phases
#     rand_dist = np.random.uniform(0, 2*np.pi, size=amps.shape)
#     rand_phases = np.exp(1j * rand_dist)
# 
#     # Multiply phases to amplitudes and create power law
#     amps *= rand_phases
#     amps /= freqs ** (slope / 2)
# 
#     # Add oscillations
#     amps_osc = amps.copy()
#     if periodic_params:
#         for osc_params in periodic_params:
#             freq_osc, amp_osc, width = osc_params
#             amp_dist = sp.stats.norm(freq_osc, width).pdf(freqs)
#             # add same random phases
#             amp_dist = amp_dist * rand_phases
#             amps_osc += amp_osc * amp_dist
# 
#     # Create colored noise time series from amplitudes
#     noise = irfft(amps)
#     noise_osc = irfft(amps_osc)
# 
#     # Add white noise
#     if nlv:
#         w_noise = np.random.normal(scale=nlv, size=n_samples-2)
#         noise += w_noise
#         noise_osc += w_noise
# 
#     # Highpass filter
#     if highpass:
#         sos = sig.butter(4, 1, btype="hp", fs=srate, output='sos')
#         noise = sig.sosfilt(sos, noise)
#         noise_osc = sig.sosfilt(sos, noise_osc)
# 
#     return noise, noise_osc
# =============================================================================


def calc_PSDs(data_EEG, pre_seconds, post_samples, seiz_len_samples, srate,
              **welch_params):
    """Calc PSDs."""
    # n_freq = srate // 2 + 1
    pre_samples = pre_seconds * srate  # 10 Seconds before seizure start

    nperseg = welch_params["nperseg"]
    n_freq = np.fft.rfftfreq(nperseg, d=1/srate).size

    psd_EEG_pre = np.zeros(n_freq)
    psd_EEG_seiz = np.zeros(n_freq)
    psd_EEG_post = np.zeros(n_freq)

    pre_slice = slice(pre_samples)
    seiz_slice = slice(pre_samples, pre_samples + seiz_len_samples)
    post_slice = slice(pre_samples + seiz_len_samples, None)

    data_pre = data_EEG[pre_slice]
    data_seiz = data_EEG[seiz_slice]
    data_post = data_EEG[post_slice]

    freq, psd_EEG_pre = sig.welch(data_pre, **welch_params)
    freq, psd_EEG_seiz = sig.welch(data_seiz, **welch_params)
    freq, psd_EEG_post = sig.welch(data_post, **welch_params)

    return freq, psd_EEG_pre, psd_EEG_seiz, psd_EEG_post


# =============================================================================
# def saw_noise(srate, pre_seconds, post_samples, seiz_len_samples, slope=2,
#               saw_power=0.9, saw_width=0.54, freq=3, duration=30, seed=1,
#               **welch_params):
#     """Make Docstring."""
#     if seed:
#         np.random.seed(seed)
#     pre_samples = pre_seconds * srate  # 10 Seconds before seizure start
# 
#     # Sawtooth signal
#     time_saw = np.arange(0, duration_seconds, 1/srate)
#     saw = sawtooth(2 * np.pi * freq * time_saw, width=saw_width)
#     # saw = saw[:-2]
#     saw *= saw_power  # scaling
# 
#     # 1/f noise
#     _, noises = osc_signals(slope, srate=srate, duration=duration_seconds)
# 
#     # Highpass 0.3 Hz like real signal
#     # sos = sig.butter(20, 1, btype="hp", fs=srate, output='sos')
#     # noises = sig.sosfilt(sos, noises)
# 
#     # make signal 10 seconds zero, 10 seconds strong, 10 seconds zero
#     psd_saw_seiz = np.r_[np.zeros(pre_samples),
#                          saw[:seiz_len_samples],
#                          np.zeros(post_samples)]
#     print("noises: ", noises.shape)
#     print("psd_saw_seiz: ", psd_saw_seiz.shape)
#     noise_saw = noises + psd_saw_seiz
# 
#     # normalize
#     noise_saw = (noise_saw - noise_saw.mean()) / noise_saw.std()
# 
#     # PSD
#     pre_slice = slice(pre_samples)
#     seiz_slice = slice(pre_samples, pre_samples + seiz_len_samples)
#     post_slice = slice(pre_samples + seiz_len_samples, None)
# 
#     freq, psd_saw_pre = sig.welch(noise_saw[pre_slice], **welch_params)
#     freq, psd_saw_seiz = sig.welch(noise_saw[seiz_slice], **welch_params)
#     freq, psd_saw_post = sig.welch(noise_saw[post_slice], **welch_params)
#     return time_saw, noise_saw, freq, psd_saw_pre, psd_saw_seiz, psd_saw_post
# =============================================================================


def osc_signals_old(samples, slopes, freq_osc=[], amp=[], width=[],
                    srate=2400, seed=1):
    """Simplified sim function."""
    if seed:
        np.random.seed(seed)
    # Initialize output
    slopes = [slopes]
    noises = np.zeros([len(slopes), samples])
    noises_pure = np.zeros([len(slopes), samples])
    # Make fourier amplitudes
    amps = np.ones(samples//2 + 1, complex)
    freqs = np.fft.rfftfreq(samples, d=1/srate)

    # Make 1/f
    freqs[0] = 1  # avoid divison by 0
    random_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)

    for j, slope in enumerate(slopes):
        # Multiply Amp Spectrum by 1/f
        # half slope needed:
        # 1/f^2 in power spectrum = sqrt(1/f^2)=1/f^2*0.5=1/f
        # in amp spectrum
        amps = amps / freqs ** (slope / 2)
        amps *= np.exp(1j * random_phases)
        noises_pure[j] = np.fft.irfft(amps)
        for i in range(len(freq_osc)):
            # make Gaussian peak
            amp_dist = sp.stats.norm(freq_osc[i], width[i]).pdf(freqs)
            # normalize peak for smaller amplitude differences for different
            # frequencies:
            amp_dist /= np.max(amp_dist)
            amps += amp[i] * amp_dist
    noises[j] = np.fft.irfft(amps)
    return noises, noises_pure


def saw_noise_old(srate, pre_seconds, post_samples, seiz_len_samples, slope=[2], saw_power=0.9,
                  saw_width=0.54, freq=3, duration=30, seed=1, **welch_params):
    """Make Docstring."""
    if seed:
        np.random.seed(seed)
    # duration and sampling
    time = np.arange(duration_samples)
    samples = time.size
    pre_samples = pre_seconds * srate  # 10 Seconds before seizure start

    # Sawtooth signal
    time_saw = np.arange(0, duration_seconds, 1/srate)
    saw = sawtooth(2 * np.pi * freq * time_saw, width=saw_width)
    saw = saw[:-2]
    saw *= saw_power  # scaling

    # 1/f noise
    noises, _ = osc_signals_old(samples, slope)
    noises = noises[0]

    # Highpass 0.3 Hz like real signal
    # sos = sig.butter(20, 1, btype="hp", fs=srate, output='sos')
    # noises = sig.sosfilt(sos, noises)

    # make signal 10 seconds zero, 10 seconds strong, 10 seconds zero
    psd_saw_seiz = np.r_[np.zeros(pre_samples),
                         saw[:seiz_len_samples],
                         np.zeros(post_samples)]

    noise_saw = noises + psd_saw_seiz

    # normalize
    noise_saw = (noise_saw - noise_saw.mean()) / noise_saw.std()

    # PSD
    pre_slice = slice(pre_samples)
    seiz_slice = slice(pre_samples, pre_samples+seiz_len_samples)
    post_slice = slice(pre_samples + seiz_len_samples, None)

    freq, psd_saw_pre = sig.welch(noise_saw[pre_slice], **welch_params)
    freq, psd_saw_seiz = sig.welch(noise_saw[seiz_slice], **welch_params)
    freq, psd_saw_post = sig.welch(noise_saw[post_slice], **welch_params)
    return time_saw, noise_saw, freq, psd_saw_pre, psd_saw_seiz, psd_saw_post


def fooof_fit(psd: np.array, cond: str, freq: np.array,
              freq_range: tuple, fooof_params: dict) -> tuple:
    """
    Return aperiodic fit and corresponding label.

    Parameters
    ----------
    psd : np.array
        PSD.
    cond : str
        Condition.
    freq : np.array
        Freq array for PSD.
    freq_range : tuple of int
        Fitting range.
    fooof_params : dict
        Fooof params.

    Returns
    -------
    tuple(ndarray, str)
        (aperiodic fit, plot label).
    """
    fm = FOOOF(**fooof_params)
    fm.fit(freq, psd, freq_range)
    exp = fm.get_params("aperiodic", "exponent")
    label = f"1/f {cond}={exp:.2f}"
    ap_fit = gen_aperiodic(freq, fm.aperiodic_params_)
    return 10**ap_fit, label


# %% Parameters

# Paths
data_path = "../data/Fig3/"
fig_path = "../paper_figures/"
fig_name = "Fig3_f_Separation.pdf"
fig_name_supp = "Fig3_f_Separation_SuppMat.pdf"

# Colors
c_empirical = "purple"
c_sim = "k"

c_pre = "c"
c_seiz = "r"
c_post = "y"

# EEG Params
srate = 256
cha_nm = "F3-C3"

"""
STRATEGIE: separiere daten in pre-, seiz, und post und eliminiere die Zeitpunkte
berechne PSDS separat, füge sie im plot manuell zusammen.
Das Gleche mit saw signal.
"""

# Seizure sample timepoints
seiz_start_samples = 87800  # behalten
seiz_end_samples = 91150  # behalten
seiz_len_samples = seiz_end_samples - seiz_start_samples  # behalten


# Pre- and post-seizure evaluation:
pre_seconds = 10  # start evaluation 10 seconds before seizure start  # behalten
duration_seconds = 3 * pre_seconds# + seiz_len_samples / srate # plot time series  # eliminate
duration_samples = duration_seconds * srate

####
#duration_samples = duration_seconds * srate  # plot time series
post_samples = 2*pre_seconds*srate - seiz_len_samples

pre_seiz_samples = int(seiz_start_samples - pre_seconds*srate)
post_seiz_samples = int(seiz_end_samples + pre_seconds*srate)

post_seconds = seiz_start_samples / srate + pre_seconds# + duration_seconds

# Welch Params
nperseg = srate
welch_params = {"fs": srate, "nperseg": nperseg}

# Fooof params: standard
fooof_params = dict(verbose=False)

# %% Get data

# Seizure sample timepoints
#seiz_start_samples = 87800  # behalten
#seiz_end_samples = 91150  # behalten
# seiz_len_samples = seiz_end_samples - seiz_start_samples  # behalten
# Load data
seiz_data = np.load(data_path + cha_nm + ".npy", allow_pickle=True)
time_EEG = np.linspace(0, seiz_data.size//srate, num=seiz_data.size)

# Select seizure time points
seiz_data = seiz_data[pre_seiz_samples:post_seiz_samples]
time_EEG = time_EEG[pre_seiz_samples:post_seiz_samples]

np.save(data_path + cha_nm + "_short.npy", seiz_data)


# Load data
data_EEG = np.load(data_path + cha_nm + "_short.npy", allow_pickle=True)
time_EEG = np.linspace(0, data_EEG.size//srate, num=data_EEG.size)

# Select seizure time points
seiz_start_samples = 10 * srate
seiz_len_samples = 3350
seiz_end_samples = seiz_start_samples + seiz_len_samples

# Calc pre, post, and seizure PSDs
EEG_psds = calc_PSDs(data_EEG, pre_seconds, post_samples, seiz_len_samples,
                     srate, **welch_params)
freq, psd_EEG_pre, psd_EEG_seiz, psd_EEG_post = EEG_psds

# Sawtooth Signal
saw_power = 0.004
saw_width = 0.69
seed = 2
slope = 1.8
saw_params = dict(slope=slope, saw_power=saw_power, saw_width=saw_width,
                  seed=seed, duration=duration_seconds)

# Simulate saw tooth signal and calc PSDs (inside the function)
saw_psds = saw_noise_old(srate, pre_seconds, post_samples, seiz_len_samples,
                         **saw_params)
time_saw, noise_saw, _, psd_saw_pre, psd_saw_seiz, psd_saw_post = saw_psds
# %% Fit

# Calc fooof pre-, post-, and during seizure
freq_range = [1, 100]
calc_fooof = dict(freq=freq, freq_range=freq_range, fooof_params=fooof_params)

fit_pre_eeg, lab_pre_eeg = fooof_fit(psd_EEG_pre, "Pre", **calc_fooof)
fit_seiz_eeg, lab_seiz_eeg = fooof_fit(psd_EEG_seiz, "Seizure", **calc_fooof)
fit_post_eeg, lab_post_eeg = fooof_fit(psd_EEG_post, "Post", **calc_fooof)
fit_pre_sim, lab_pre_saw = fooof_fit(psd_saw_pre, "Pre", **calc_fooof)
fit_seiz_sim, lab_seiz_saw = fooof_fit(psd_saw_seiz, "Seizure", **calc_fooof)
fit_post_sim, lab_post_saw = fooof_fit(psd_saw_post, "Post", **calc_fooof)

# %% Plot params

width = 7.25  # inches
panel_fontsize = 14
legend_fontsize = 10
label_fontsize = 14
tick_fontsize = 14
annotation_fontsize = tick_fontsize

mpl.rcParams['xtick.labelsize'] = tick_fontsize
mpl.rcParams['ytick.labelsize'] = tick_fontsize
mpl.rcParams['axes.labelsize'] = label_fontsize
mpl.rcParams['legend.fontsize'] = legend_fontsize
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

# Tick params
ticks_time = dict(length=6, width=1.5)
ticks_psd = dict(length=4, width=1)
panel_labels = dict(x=0, y=1.01, fontsize=panel_fontsize,
                    fontdict=dict(fontweight="bold"))

# Plot Time Series
"""Exhange all numbers with meaningfull variable variables"""
# a
# EEG Time Series
x_step_seconds = 5  # seconds
x_step_samples = x_step_seconds * srate
xticks_a = time_EEG[::x_step_samples]
xticklabels_a = []
yticks_a = [-200, 0, 200]
yticklabels_a = [-200, "", 200]
xlim_a = [0, None]
ylabel_a = fr"{cha_nm} [$\mu$V]"

axes_a = dict(xticks=xticks_a, xticklabels=xticklabels_a,
              yticks=yticks_a, yticklabels=yticklabels_a, xlim=xlim_a)

# Rectangles to mark pre-, seizure, post
rect_min = data_EEG.min() * 1.1
rect_height = np.abs(data_EEG).max() * 2
rect = dict(height=rect_height, alpha=0.2)

start_pre = 0
start_seiz = seiz_start_samples / srate
start_post = seiz_end_samples / srate

xy_pre = (start_pre, rect_min)
xy_seiz = (start_seiz, rect_min)
xy_post = (start_post, rect_min)

width_pre = pre_seconds
width_seiz = seiz_len_samples / srate
width_post = pre_seconds

rect_EEG_pre_params = dict(xy=xy_pre, width=width_pre, color=c_pre, **rect)
rect_EEG_seiz_params = dict(xy=xy_seiz, width=width_seiz, color=c_seiz, **rect)
rect_EEG_post_params = dict(xy=xy_post, width=width_post, color=c_post, **rect)

rect_EEG_pre = plt.Rectangle(**rect_EEG_pre_params)
rect_EEG_seiz = plt.Rectangle(**rect_EEG_seiz_params)
rect_EEG_post = plt.Rectangle(**rect_EEG_post_params)

# Sawtooth Time Series
xticks_c = np.arange(0, duration_seconds + x_step_seconds, x_step_seconds)
xticklabels_c = xticks_c - pre_seconds
y_max = np.abs(noise_saw).max() * 1.1
yticks_c = (-y_max, 0, y_max)
yticklabels_c = []
xlim_c = (0, duration_seconds)
ylim_c = (yticks_c[0], yticks_c[-1])
xlabel_c = "Time [s]"
ylabel_c = "Simulation [a.u.]"

axes_c = dict(xticks=xticks_c, xticklabels=xticklabels_c,
              yticks=yticks_c, yticklabels=yticklabels_c,
              xlim=xlim_c, ylim=ylim_c, xlabel=xlabel_c)

# Rectangles to mark pre-, seizure, post
rect_min = ylim_c[0]
rect_height = np.diff(ylim_c)[0]
rect = dict(height=rect_height, alpha=0.2)

# Rect Coords
start_pre = 0
start_seiz = pre_seconds
start_post = pre_seconds + seiz_len_samples / srate

xy_pre = (start_pre, rect_min)
xy_seiz = (start_seiz, rect_min)
xy_post = (start_post, rect_min)

width_pre = pre_seconds
width_seiz = seiz_len_samples / srate
width_post = 2*pre_seconds - seiz_len_samples / srate

rect_saw_pre_params = dict(xy=xy_pre, width=width_pre, color=c_pre, **rect)
rect_saw_seiz_params = dict(xy=xy_seiz, width=width_seiz, color=c_seiz, **rect)
rect_saw_post_params = dict(xy=xy_post, width=width_post, color=c_post, **rect)

rect_saw_pre = plt.Rectangle(**rect_saw_pre_params)
rect_saw_seiz = plt.Rectangle(**rect_saw_seiz_params)
rect_saw_post = plt.Rectangle(**rect_saw_post_params)

# b
xticks_b = [1, 10, 100]
xticklabels_b = []
yticks_b = [1e-2, 1, 1e2, 1e4]
yticklabels_b = [r"$10^{-2}$", "", "", r"$10^4$"]
xlim_b = freq_range
ylabel_b = r"PSD [$\mu$$V^2$/Hz]"
xlabel_b = ""

axes_b = dict(xticks=xticks_b, xticklabels=xticklabels_b, yticks=yticks_b,
              yticklabels=yticklabels_b, xlim=xlim_b, xlabel=xlabel_b,
              ylabel=ylabel_b)

# d
yticks_d = [1e-3, 1e-1, 1e1, 1e3]
yticklabels_d = []
xlabel_d = "Frequency [Hz]"
ylabel_d = "PSD [a.u.]"
axes_d = dict(xticks=xticks_b, xticklabels=xticks_b, yticks=yticks_d,
              yticklabels=yticklabels_d, xlim=xlim_b, xlabel=xlabel_d)

# %% Plot

fig, axes = plt.subplots(2, 2,  figsize=[12, 6],
                         gridspec_kw=dict(width_ratios=[1, .6]))

# a
# Plot EEG seizure
ax = axes[0, 0]
ax.plot(time_EEG, data_EEG, c=c_empirical, lw=1)

# Set axes
ax.set(**axes_a)
ax.set_ylabel(ylabel_a, labelpad=-20)
ax.tick_params(**ticks_time)

# Add colored rectangles
ax.add_patch(rect_EEG_pre)
ax.add_patch(rect_EEG_seiz)
ax.add_patch(rect_EEG_post)


# c
# Sawtooth Time Series
ax = axes[1, 0]
ax.plot(time_saw, noise_saw, c=c_sim, lw=1)

# Add colored rectangles
ax.add_patch(rect_saw_pre)
ax.add_patch(rect_saw_seiz)
ax.add_patch(rect_saw_post)

# Set axes
ax.set(**axes_c)
ax.set_ylabel(ylabel_c, labelpad=15)
ax.tick_params(**ticks_time)


# b
# Plot EEG PSD
ax = axes[0, 1]
ax.loglog(freq, psd_EEG_pre, c_pre, lw=2)
ax.loglog(freq, psd_EEG_seiz, c_seiz, lw=2)
ax.loglog(freq, psd_EEG_post, c_post, lw=2)

# Plot EEG fooof fit
ax.loglog(freq, fit_pre_eeg, "--", c=c_pre, lw=2, label=lab_pre_eeg)
ax.loglog(freq, fit_seiz_eeg, "--", c=c_seiz, lw=2, label=lab_seiz_eeg)
ax.loglog(freq, fit_post_eeg, "--", c=c_post, lw=2, label=lab_post_eeg)

# Set axes
ax.set(**axes_b)
ax.legend()
ax.set_ylabel(ylabel_b, labelpad=-25)
ax.tick_params(**ticks_psd)


# d
# Plot saw PSD
ax = axes[1, 1]
ax.loglog(freq, psd_saw_pre, c_pre, lw=2)
ax.loglog(freq, psd_saw_seiz, c_seiz, lw=2)
ax.loglog(freq, psd_saw_post, c_post, lw=2)

# Plot Saw fooof fit
ax.loglog(freq, fit_pre_sim, "--", c=c_pre, lw=2, label=lab_pre_saw)
ax.loglog(freq, fit_seiz_sim, "--", c=c_seiz, lw=2, label=lab_seiz_saw)
ax.loglog(freq, fit_post_sim, "--", c=c_post, lw=2, label=lab_post_saw)

# Set axes
ax.set(**axes_d)
ax.legend()
ax.set_ylabel(ylabel_d, labelpad=10)
ax.tick_params(**ticks_psd)

# panel labels
for s, ax in zip("abcd", axes.flat):
    ax.text(s=s, **panel_labels, transform=ax.transAxes)

plt.tight_layout()
# plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()


# =============================================================================
# # %% Plot Supp. b)
#
# # best fooof params for seizure:
# fooof_seiz_params = dict(peak_width_limits=(0, 1), peak_threshold=0)
#
# # Message:
# # even with these tuned params the fit is above 2 and therefore bad
# # furthermore, one cannot tune fooof for each condition and then compare
# # the 1/f
#
# fm_seiz = FOOOF()
# fm_saw = FOOOF()
# fm_tuned_seiz = FOOOF(**fooof_seiz_params)
# fm_tuned_saw = FOOOF(**fooof_seiz_params)
#
# freq_range = [1, 100]
#
# fm_seiz.fit(freq, psd_EEG_seiz, freq_range)
# fm_saw.fit(freq, psd_saw_seiz, freq_range)
# fm_tuned_seiz.fit(freq, psd_EEG_seiz, freq_range)
# fm_tuned_saw.fit(freq, psd_saw_seiz, freq_range)
#
# # %% Plot Supp
# fig, axes = plt.subplots(2, 2, figsize=[12, 12], sharey="col")
# ax = axes[0, 0]
#
# fm_seiz.plot(ax=ax, plt_log=True)
# exp_seiz = fm_seiz.get_params("aperiodic", "exponent")
# ax.set_title(f"Seizure a={exp_seiz:.2f}")
# ax.set_ylabel("Fooof default parameters")
# ax.grid(False)
#
# ax = axes[0, 1]
#
# fm_saw.fit(freq, psd_saw_seiz, freq_range)
# fm_saw.plot(ax=ax, plt_log=True)
# exp_seiz = fm_saw.get_params("aperiodic", "exponent")
# ax.set_title(f"1/f={exp_seiz:.2f}")
# ax.grid(False)
# ax.set_ylabel("")
#
# ax = axes[1, 0]
#
# fm_tuned_seiz.plot(ax=ax, plt_log=True)
# exp_seiz = fm_tuned_seiz.get_params("aperiodic", "exponent")
# ax.set_title(f"1/f={exp_seiz:.2f}")
# ax.set_ylabel("Fooof tuned parameters")
# ax.grid(False)
#
# ax = axes[1, 1]
#
# fm_tuned_saw.plot(ax=ax, plt_log=True)
# exp_seiz = fm_tuned_saw.get_params("aperiodic", "exponent")
# ax.set_title(f"1/f={exp_seiz:.2f}")
# ax.set_ylabel("")
# ax.grid(False)
#
# plt.suptitle(f"Fooof fit {freq_range[0]}-{freq_range[1]}Hz  sawtooth signal")
# # plt.savefig(fig_path + fig_name_supp, bbox_inches="tight")
# plt.tight_layout()
# plt.show()
#
# =============================================================================
