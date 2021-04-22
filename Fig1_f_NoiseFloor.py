"""
Figure 2 Panels.

A: 1/f simulated with and without noise floor.
Fitted with fooof with and without knee.
Print ground truth and fit values.

B: Real spectra. Subj 12 no oscillations, maybe a subj with oscillations in
that range and an potentially early noise floor too,
maybe a full spectrum with low noise amplifier.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import mne
from pathlib import Path
from noise_helper import noise_white, osc_signals
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic


def detect_noise_floor(freq, psd, f_start=1, f_range=50, thresh=0.05):
    """
    Detect the onset of the noise floor.

    The noise floor is defined by having a slopes below 0.05.

    Parameters
    ----------
    freq : ndarray
        Freq array.
    psd : ndarray
        PSD array.
    f_start : float
        Starting frequency for the search.

    Returns
    -------
    n_start : float
        Onset frequency of the noise floor.
    """
    n_start = f_start - 1
    exp = 1
    while exp > thresh:
        n_start += 1
        fm = FOOOF(max_n_peaks=1, verbose=False)
        fm.fit(freq, psd, [n_start, n_start + f_range])
        exp = fm.get_params('aperiodic_params', 'exponent')
    return n_start + f_range // 2


# %% PARAMETERS

# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slope = 2
nperseg = srate  # welch

# Load data
f = np.load("../../Litvak11/2021-03-03_Wrap_Up/data/rest/PSD_arrays/freqs.npy")
psd_on = np.load("../../Litvak11/2021-03-03_Wrap_Up/data/rest/PSD_arrays/on_psd.npy")
psd_off = np.load("../../Litvak11/2021-03-03_Wrap_Up/data/rest/PSD_arrays/off_psd.npy")

ch_names = ['SMA', 'leftM1', 'rightM1',
            'STN_R01', 'STN_R12', 'STN_R23',
            'STN_L01', 'STN_L12', 'STN_L23']


# Save Path
fig_path = "../paper_figures/Fig1_f_NoiseFloor.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# %% Make noise

freq_range = [1, 100]

# No oscillations
freq_osc, amp = [23, 360], [0, 0]

# Make noise
w_noise = noise_white(samples-2)

pink2 = osc_signals(samples, 2, freq_osc, amp)
pink_white2 = pink2 + .13 * w_noise

freq, psd2_noise = sig.welch(pink_white2, fs=srate, nperseg=nperseg)

# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq, psd2_noise = freq[filt], psd2_noise[filt]

# Normalize
psd2_noise /= psd2_noise.max()


# %% Plot without knee

upper_limits = [10, 50, 100, 200]
alphas = [1, 0.6, .4, .2]

ap_fit_ground = gen_aperiodic(freq, np.array([0, slope]))


fig, axes = plt.subplots(1, 3, figsize=[15, 4])

ax = axes[0]

# Detect Noise floor
floor = detect_noise_floor(freq, psd2_noise)
signal = (freq <= floor)
noise = (freq >= floor)

ax.loglog(freq[signal], psd2_noise[signal], "k", label="Spectrum")
ax.loglog(freq[noise], psd2_noise[noise], "darkgray", label="Noise floor")
ax.loglog(freq, 10**ap_fit_ground, "k--", label=f"Ground truth a={slope}")

for i, lim in enumerate(upper_limits):
    fm = FOOOF(verbose=False)
    fm.fit(freq, psd2_noise, [1, lim])
    exp = fm.get_params('aperiodic_params', 'exponent')
    ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
    ax.loglog(fm.freqs, 10**ap_fit, "b-", lw=2, alpha=alphas[i],
              label=f"1-{lim}Hz a={exp:.2f}")

xlim = 1, 600
offset = psd2_noise[freq == xlim[1]][0]
ylim = .5 * offset, 1

ax.set_ylim(ylim)
ax.set_xlim(xlim)
ax.legend()

xticks = [1] + upper_limits + [600]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ax.set_xlabel("Frequency in Hz")
ax.set_ylabel("Normalized power")


ax = axes[1]

psd_lfp = psd_off[11, 8]
psd_lfp_osc = psd_off[8, 3]

floor = detect_noise_floor(freq, psd_lfp)
signal = (freq <= floor)
noise = (freq >= floor)
ax.loglog(freq[signal], psd_lfp[signal], "k", label="Subj. 12 STN-L23 Off")
ax.loglog(freq[noise], psd_lfp[noise], "darkgray")

floor = detect_noise_floor(freq, psd_lfp_osc)
signal = (freq <= floor)
noise = (freq >= floor)
ax.loglog(freq[signal], psd_lfp_osc[signal], "k", label="Subj. 9 STN-R01 Off")
ax.loglog(freq[noise], psd_lfp_osc[noise], "darkgray", label="Noise floor")

# ax.vlines(23, 0, 1) # check peaks
# ax.vlines(360, 0, 1)

ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ax.set_xlabel("Frequency in Hz")
ax.set_ylabel(r"Power in $\mu$V/Hz")
ax.legend()

ax = axes[2]

# No oscillations
freq_osc, amp, width = [20, 350], [10, 10], [1, 1]

# Make noise
w_noise = noise_white(samples-2)

pink2 = osc_signals(samples, 2, freq_osc, amp, width)
pink_white2 = pink2 + .13 * w_noise

freq, psd2_noise_osc = sig.welch(pink_white2, fs=srate, nperseg=nperseg)

# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq, psd2_noise_osc = freq[filt], psd2_noise_osc[filt]

# Normalize
psd2_noise_osc /= psd2_noise_osc.max()

floor = detect_noise_floor(freq, psd2_noise_osc)
signal = (freq <= floor)
noise = (freq >= floor)
ax.loglog(freq[signal], psd2_noise_osc[signal], "k", label="Subj. 12 STN-L23 Off")
ax.loglog(freq[noise], psd2_noise_osc[noise], "darkgray")

plt.show()


"""
Eliminate grey rectangles. Choose colors for spectrum, the start of the noise
floor changes the color of the spectrum to black or grey

Message: noise floor difficult to detect. If 300Hz oscillation power increases:
    -> noise floor earlier
    if beta power increases:
    -> noise floor later
Add simulation with beta peak and 300 peak and vary power to show noise floor
shift. 
Where -> C
"""


# %%
fig, ax = plt.subplots(1, 1)

# No oscillations
freq_osc, amp, width = [23, 360], [10, 50], [5, 50]

# Make noise
w_noise = noise_white(samples-2)

pink2 = osc_signals(samples, 2, freq_osc, amp, width)
pink_white2 = pink2 + .13 * w_noise

freq, psd2_noise_osc = sig.welch(pink_white2, fs=srate, nperseg=nperseg)

# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq, psd2_noise_osc = freq[filt], psd2_noise_osc[filt]

# Normalize
psd2_noise_osc /= psd2_noise_osc.max()

floor = detect_noise_floor(freq, psd2_noise_osc)
signal = (freq <= floor)
noise = (freq >= floor)
ax.loglog(freq[signal], psd2_noise_osc[signal], "k", label="Subj. 12 STN-L23 Off")
ax.loglog(freq[noise], psd2_noise_osc[noise], "darkgray")

plt.show()





# =============================================================================
# # %% Plot knee
# upper_limits = [100, 200, 400, 1200]
# 
# 
# fig, ax = plt.subplots(1, 1, figsize=[5, 4])
# 
# ax.loglog(freq, psd2_noise, "k", label="Spectrum", lw=3)
# ax.loglog(freq, 10**ap_fit_ground, "k--", lw=1,
#           label=f"Ground truth a={slope}")
# 
# for i, lim in enumerate(upper_limits):
#     fm = FOOOF(verbose=False, max_n_peaks=0, aperiodic_mode="knee")
#     fm.fit(freq, psd2_noise, [1, lim])
#     exp = fm.get_params('aperiodic_params', 'exponent')
#     ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
#     ax.loglog(fm.freqs, 10**ap_fit, "b-", lw=3, alpha=alphas[i],
#               label=f"1-{lim}Hz a={exp:.2f}")
# # Detect Noise floor
# floor = detect_noise_floor(freq, psd2_noise, 120)
# 
# xlim = 1, 400
# ylim = 5e-4, 1
# offset = psd2_noise[freq == xlim[1]][0]
# rec_h = plt.Rectangle((xlim[0], ylim[0]), np.diff(xlim)[0],
#                       offset - ylim[0], color="grey", alpha=0.2)
# rec_v = plt.Rectangle((123, offset), xlim[1]-123,
#                       np.diff(ylim)[0], color="grey", alpha=0.2)
# ax.add_artist(rec_h)
# ax.add_artist(rec_v)
# 
# ax.set_ylim(ylim)
# ax.set_xlim(xlim)
# handles, labels = ax.get_legend_handles_labels()
# handles.append(rec_h)
# labels.append("Noise floor")
# ax.legend(handles, labels)
# 
# xticks = [1] + upper_limits
# ax.set_xticks(xticks)
# ax.set_xticklabels(xticks)
# ax.set_xlabel("Frequency in Hz")
# ax.set_ylabel("Power")
# ax.set_title("Knee doesn't help")
# plt.show()
# 
# =============================================================================
