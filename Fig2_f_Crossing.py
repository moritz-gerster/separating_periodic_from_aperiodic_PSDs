"""
Message: Fitting range borders must be 100% oscillations free.

Plan B:

Panel A: Oscillations crossing different fitting borders
below: fitting error.

In practice diffucult to avoid: Delta oscillations, show sim.


Plan A:

Panel A: Oscillations crossing different fitting borders
below: fitting error. check

Panel B show subj 10 huge beta:
    peak extends fitting range: 30-45Hz, 40-60Hz? 1-100Hz? 1-40?
        strong impact of delta offset: 1-100, 1-40

(other subs:
    40-60Hz very flat, strong impact by noise: don't fit 40-60Hz if a<1?)

Supp. Mat. 2b): Show fooof fits

Panel C: Show simulation of this spectrum.
    Show 30-50 wrong
    Show noise dependence of 1/f
    Show delta error

In practice diffucult to avoid: Delta oscillations, show sim.

Add own algorithm: fit straight line between 1Hz and upper fitting range value

Add plots of the time series??
"""


import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

from pathlib import Path
#import pandas as pd
from fooof import FOOOF, FOOOFGroup
from noise_helper import noise_white, osc_signals, psds_pink, slope_error
from noise_helper import plot_all, irasa, osc_signals_correct
import mne
from mne.time_frequency import psd_welch
from fooof.sim.gen import gen_aperiodic

import seaborn as sns
sns.set()

# %% PARAMETERS

# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slopes = np.arange(1, 4.5, 1)

# WELCH
nperseg = srate  # 4*srate too high resolution for fooof

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig2_f_crossing.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# %% Make signals and fit
up = 100
lower = np.arange(0, 80, 1)

freq_osc = [4, 11, 23]  # Hz
amp = [30, 30, 10]
width = [0.1, 1, 2]

signals = osc_signals(samples, slopes, freq_osc, amp, width=width)
freq, noise_psds = psds_pink(signals, srate, nperseg)

# Filter 1-600Hz
freq, noise_psds = freq[1:600], noise_psds[:, 1:600]

# Normalize
noise_psds /= noise_psds[:, 0, None]

# Calc fooof vary freq ranges
errors = []
for low in lower:
    fm = FOOOFGroup(verbose=None)
    fm.fit(freq, noise_psds, [low, up])
    exp = fm.get_params("aperiodic", "exponent")
    error = slopes - exp
    error = np.sum(np.abs(error))
    errors.append(error)

# %% Panel A
fig, axes = plt.subplots(2, 1, figsize=[6, 6], sharex=True)

ax = axes[0]

mask = (freq <= 100)
for i in range(slopes.size):
    ax.loglog(freq[mask], noise_psds[i, mask], "k",
              label=f"1/f={slopes[i]:.2f}")
ylim = ax.get_ylim()
ax.vlines(100, *ylim, color="k", label="Upper fitting range border")
# =============================================================================
# for f in freq_osc:
#     ax.vlines(f, *ylim, label="Lower fitting range border")
# =============================================================================

ax.set_ylabel("Normalized power")
# ax.legend()

ax = axes[1]
ax.plot(lower, errors)
ax.set_ylabel("1/f Error")
ax.set_xlabel("Frequency in Hz")
plt.show()


# %% Panel B
ch = 'STN_L23'

# Load data
path = "../data/Fig2/"
fname10_off = "subj10_off_R14_raw.fif"
fname10_on = "subj10_on_R8_raw.fif"

sub10_off = mne.io.read_raw_fif(path + fname10_off, preload=True)
sub10_on = mne.io.read_raw_fif(path + fname10_on, preload=True)

sub10_off.pick_channels(ch)
sub10_on.pick_channels(ch)

filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}

sub10_off.notch_filter(**filter_params)
sub10_on.notch_filter(**filter_params)

welch_params = {"fmin": 1,
                "fmax": 600,
                "tmin": 0.5,
                "tmax": 185,
                "n_fft": srate,
                "n_overlap": srate // 2,
                "average": "mean"}

spec10_off, freq = psd_welch(sub10_off, **welch_params)
spec10_on, freq = psd_welch(sub10_on, **welch_params)

spec10_on, spec10_off = spec10_on[0], spec10_off[0]
# %% Plot fooof fit in spectrum

frange1 = [1, 95]
frange2 = [30, 45]
frange3 = [40, 60]
frange4 = [1, 45]
frange5 = [1, 120]

fm1_on = FOOOF(peak_width_limits=[1, 100])
fm1_on.fit(freq, spec10_on, frange1)
ap_params1_on = fm1_on.get_params("aperiodic")
exp1_on = fm1_on.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit1_on = gen_aperiodic(fm1_on.freqs, fm1_on.aperiodic_params_)

fm1_off = FOOOF(peak_width_limits=[1, 100])
fm1_off.fit(freq, spec10_off, frange1)
ap_params1_off = fm1_off.get_params("aperiodic")
exp1_off = fm1_off.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit1_off = gen_aperiodic(fm1_off.freqs, fm1_off.aperiodic_params_)

fm2_on = FOOOF(peak_width_limits=[1, 100])
fm2_on.fit(freq, spec10_on, frange2)
ap_params1_on = fm2_on.get_params("aperiodic")
exp2_on = fm2_on.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit2_on = gen_aperiodic(fm2_on.freqs, fm2_on.aperiodic_params_)

fm2_off = FOOOF(peak_width_limits=[1, 100])
fm2_off.fit(freq, spec10_off, frange2)
ap_params1_off = fm2_off.get_params("aperiodic")
exp2_off = fm2_off.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit2_off = gen_aperiodic(fm2_off.freqs, fm2_off.aperiodic_params_)


fm3_on = FOOOF(max_n_peaks=0)
fm3_on.fit(freq, spec10_on, frange3)
ap_params1_on = fm3_on.get_params("aperiodic")
exp3_on = fm3_on.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit3_on = gen_aperiodic(fm3_on.freqs, fm3_on.aperiodic_params_)

fm3_off = FOOOF(max_n_peaks=0)
fm3_off.fit(freq, spec10_off, frange3)
ap_params1_off = fm3_off.get_params("aperiodic")
exp3_off = fm3_off.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit3_off = gen_aperiodic(fm3_off.freqs, fm3_off.aperiodic_params_)


fm4_on = FOOOF(peak_width_limits=[1, 100])
fm4_on.fit(freq, spec10_on, frange4)
ap_params1_on = fm4_on.get_params("aperiodic")
exp4_on = fm4_on.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit4_on = gen_aperiodic(fm4_on.freqs, fm4_on.aperiodic_params_)

fm4_off = FOOOF(peak_width_limits=[1, 100])
fm4_off.fit(freq, spec10_off, frange4)
ap_params1_off = fm4_off.get_params("aperiodic")
exp4_off = fm4_off.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit4_off = gen_aperiodic(fm4_off.freqs, fm4_off.aperiodic_params_)


fm5_on = FOOOF(peak_width_limits=[1, 100])
fm5_on.fit(freq, spec10_on, frange5)
ap_params1_on = fm5_on.get_params("aperiodic")
exp5_on = fm5_on.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit5_on = gen_aperiodic(fm5_on.freqs, fm5_on.aperiodic_params_)

fm5_off = FOOOF(peak_width_limits=[1, 100])
fm5_off.fit(freq, spec10_off, frange5)
ap_params1_off = fm5_off.get_params("aperiodic")
exp5_off = fm5_off.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit5_off = gen_aperiodic(fm5_off.freqs, fm5_off.aperiodic_params_)

# %% Panel B

fig, axes = plt.subplots(1, 1, figsize=[8, 8])

# ax = axes[0]
ax = axes

ax.loglog(freq, spec10_on, label=ch + " on")
ax.loglog(fm1_on.freqs, 10**ap_fit1_on, label=f"{frange1}Hz a={exp1_on:.2f} On")
ax.loglog(fm2_on.freqs, 10**ap_fit2_on, label=f"{frange2}Hz a={exp2_on:.2f} On")
ax.loglog(fm3_on.freqs, 10**ap_fit3_on, label=f"{frange3}Hz a={exp3_on:.2f} On")
ax.loglog(fm4_on.freqs, 10**ap_fit4_on, label=f"{frange4}Hz a={exp4_on:.2f} On")
# ax.loglog(fm5_on.freqs, 10**ap_fit5_on, label=f"{frange5}Hz a={exp4_on:.2f} On")
ax.legend()

# =============================================================================
# ax = axes[1]
# 
# ax.loglog(freq, spec10_off, label=ch + " Off")
# ax.loglog(fm1_off.freqs, 10**ap_fit1_off,
#           label=f"{frange1}Hz a={exp1_off:.2f} Off")
# ax.loglog(fm2_off.freqs, 10**ap_fit2_off,
#           label=f"{frange2}Hz a={exp2_off:.2f} Off")
# ax.loglog(fm3_off.freqs, 10**ap_fit3_off,
#           label=f"{frange3}Hz a={exp3_off:.2f} Off")
# ax.loglog(fm4_off.freqs, 10**ap_fit4_off,
#           label=f"{frange4}Hz a={exp4_off:.2f} Off")
# ax.loglog(fm5_off.freqs, 10**ap_fit5_off,
#           label=f"{frange5}Hz a={exp5_off:.2f} Off")
# ax.legend()
# =============================================================================
plt.show()

# %% Supp Mat Panel B: Check fooof

fig, axes = plt.subplots(2, 4, figsize=[16, 8])

ax = axes[0, 0]
fm1_on.plot(ax=ax, plt_log=True, add_legend=False)
ax.set_title("On " + str(frange1))

ax = axes[1, 0]
fm1_on.plot(ax=ax, plt_log=False, add_legend=False)

ax = axes[0, 1]
fm2_on.plot(ax=ax, plt_log=True, add_legend=False)
ax.set_title("On " + str(frange2))

ax = axes[1, 1]
fm2_on.plot(ax=ax, plt_log=False, add_legend=False)

ax = axes[0, 2]
fm3_on.plot(ax=ax, plt_log=True, add_legend=False)
ax.set_title("On " + str(frange3))

ax = axes[1, 2]
fm3_on.plot(ax=ax, plt_log=False, add_legend=False)

ax = axes[0, 3]
fm4_on.plot(ax=ax, plt_log=True, add_legend=False)
ax.set_title("On " + str(frange4))

ax = axes[1, 3]
fm4_on.plot(ax=ax, plt_log=False, add_legend=False)
plt.show()

# %% Reproduce PSD with a sim

# No oscillations
freq_osc = [28, 360]
amp = [1000, 500000]
width = [10, 60]

# Make noise
w_noise = noise_white(samples-2)

pink3 = osc_signals(samples, 3, freq_osc, amp, width, normalize=True)
pink_steep = pink3
#pink_steep = pink3 + .165 * w_noise
#pink3 = osc_signals(samples, 3, freq_osc, amp, width, normalize=False)
#pink_steep = pink3 + .13 * w_noise

freq, sim3 = sig.welch(pink_steep, fs=srate, nperseg=nperseg)

# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq, sim3 = freq[filt], sim3[filt]

# Adjust offset for real spectrum
sim3 /= (sim3[0] / spec10_on[0])

fig, axes = plt.subplots(1, 1, figsize=[8, 8])

# ax = axes[0]
ax = axes
ax.loglog(freq, spec10_on, label=ch + " on")
ax.loglog(freq, sim3, label="Sim a=3")
ax.legend()
plt.show()

# %%
seed = int(time.time())
# np.random.seed(seed)

freq_osc, amp, width = None, None, None
# pink3 = osc_signals_correct(samples, 0, freq_osc, amp, width, normalize=True)
pink3 = osc_signals_correct(samples, [0], seed=seed)

freq, sim3 = sig.welch(pink3, fs=srate, nperseg=nperseg)
filt = (freq > 0) & (freq <= 600)
freq, sim3 = freq[filt], sim3[0, filt]

#freq, sim3 = sig.welch(w_noise, fs=srate, nperseg=nperseg)
#freq, sim3 = freq[filt], sim3[filt]

fig, axes = plt.subplots(1, 1, figsize=[8, 8])

# ax = axes[0]
ax = axes
ax.loglog(freq, sim3, label="Sim a=3")
ax.legend()
plt.show()

# %%

slope = 0

# Make fourier amplitudes
amps = np.ones(samples//2 + 1, complex)
freqs = np.fft.rfftfreq(samples, d=1/srate)

# Make 1/f
# Generate random phases
random_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)

# Multiply Amp Spectrum by 1/f
freqs[0] = 1  # avoid divison by 0

amps /= freqs ** (slope / 2)  # half slope needed
amps *= np.exp(1j * random_phases)


# Add peaks
i = 0
freq_osc = [100]
width = [10]
amp = [10000]

freq_idx = np.abs(freqs - freq_osc[i]).argmin()
# make Gaussian peak
amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
# amp_dist /= np.max(amp_dist)    # check ich nciht
amps += amp[i] * amp_dist

noises = np.fft.irfft(amps, norm="ortho")

freq, sim3 = sig.welch(noises, fs=srate, nperseg=nperseg)
filt = (freq > 0) & (freq <= 600)
freq, sim3 = freq[filt], sim3[filt]


fig, axes = plt.subplots(1, 1, figsize=[8, 8])

# ax = axes[0]
ax = axes
ax.loglog(freq, sim3, label="func")
ax.legend()
plt.show()
