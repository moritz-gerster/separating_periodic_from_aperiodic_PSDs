"""
Figure 2 Panels.

A: 1/f simulated with and without noise floor.
Fitted with fooof with and without knee.
Print ground truth and fit values.

B: Real spectra. Subj 12 no oscillations, maybe a subj with oscillations in
that range and an potentially early noise floor too,
maybe a full spectrum with low noise amplifier.

Vielleicht noch vertikale Linien für max fitting range einzeichen?
Dann leichter zu erkennen. Vielleicht Legende weglassen und a Werte direkt
in den Plot schreiben

Evt Panel C zufügen mit Simulartion wo Noise Floor verschoben wird
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import mne
from pathlib import Path
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from mne.time_frequency import psd_welch
import seaborn as sns
from scipy.stats import norm
sns.set()


def noise_white(samples, seed=True):
    """Create White Noise of N samples."""
    if seed:
        np.random.seed(10)
    noise = np.random.normal(0, 1, size=samples)
    return noise


def osc_signals(samples, slope, freq_osc, amp, width=None, seed=True,
                srate=2400, normalize=True):
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
        if normalize:
            # normalize
            return (noise - noise.mean()) / noise.std()
        else:
            return noise
    elif isinstance(slope, (np.ndarray, list)):
        pink_noises = np.zeros([len(slope), samples-2])
        for i in range(len(slope)):
            # Multiply Amp Spectrum by 1/f
            amps_i = amps / freqs ** (slope[i] / 2)  # half slope needed
            amps_i *= np.exp(1j * random_phases)
            # Transform back to get pink noise time series
            noise = np.fft.irfft(amps_i)
            if normalize:
                # normalize
                pink_noises[i] = (noise - noise.mean()) / noise.std()
            else:
                pink_noises[i] = noise
        return pink_noises



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
path = "../data/Fig1/"
fname12 = "subj12_off_R7_raw.fif"
fname9 = "subj9_off_R1_raw.fif"

sub12 = mne.io.read_raw_fif(path + fname12, preload=True)
sub9 = mne.io.read_raw_fif(path + fname9, preload=True)

sub12.pick_channels(['STN_L23'])
sub9.pick_channels(['STN_R01'])

filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}

sub12.notch_filter(**filter_params)
sub9.notch_filter(**filter_params)

welch_params = {"fmin": 1,
                "fmax": 600,
                "tmin": 0.5,
                "tmax": 185,
                "n_fft": srate,
                "n_overlap": srate // 2,
                "average": "mean"}

spec12, freq = psd_welch(sub12, **welch_params)
spec9, freq = psd_welch(sub9, **welch_params)

psd_lfp = spec12[0]
psd_lfp_osc = spec9[0]

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig1_f_NoiseFloor.pdf"
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
alphas = [1, 0.7, .5, .3]

ap_fit_ground = gen_aperiodic(freq, np.array([0, slope]))



fig, axes = plt.subplots(1, 2, figsize=[10, 4],
                         gridspec_kw=dict(width_ratios=[0.8, 1]))

ax = axes[0]

# Detect Noise floor
floor = detect_noise_floor(freq, psd2_noise)
signal = (freq <= floor)
noise = (freq >= floor)

ax.loglog(freq[signal], psd2_noise[signal], "k", label="Spectrum")
ax.loglog(freq[noise], psd2_noise[noise], "darkgray", label="Noise floor")
ax.loglog(freq, 10**ap_fit_ground, "k:", label=f"Ground truth a={slope}", lw=1)

for i, lim in enumerate(upper_limits):
    fm = FOOOF(verbose=False)
    fm.fit(freq, psd2_noise, [1, lim])
    exp = fm.get_params('aperiodic_params', 'exponent')
    ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
    ax.loglog(fm.freqs, 10**ap_fit, "-", c="blue", lw=2, alpha=alphas[i]-0.1,
              label=f"1-{lim}Hz a={exp:.2f}") # c="dodgerblue"

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

# Plot Sub 12
floor = detect_noise_floor(freq, psd_lfp)
signal = (freq <= floor)
noise = (freq >= floor)
ax.loglog(freq[signal], psd_lfp[signal], "purple", label="LFP Sub. 12") # Off STN-L23
ax.loglog(freq[noise], psd_lfp[noise], "darkgray")

# Plot Sub 9
floor_osc = detect_noise_floor(freq, psd_lfp_osc)
signal = (freq <= floor_osc)
noise = (freq >= floor_osc)
ax.loglog(freq[signal], psd_lfp_osc[signal], "purple", label="LFP Sub. 9") # Off STN-R01
ax.loglog(freq[noise], psd_lfp_osc[noise], "darkgray", label="Noise floor")

# Get peak freqs, heights, and noise heights
peak_freq1 = 23
peak_freq2 = 350

peak_height1 = psd_lfp_osc[freq == peak_freq1]
peak_height2 = psd_lfp_osc[freq == peak_freq2]

noise_height = psd_lfp[freq == floor]
noise_height_osc = psd_lfp_osc[freq == floor_osc]

# Plot Peak lines
ax.vlines(peak_freq1, noise_height_osc, peak_height1,
          color="k",
          linestyle="--", lw=1)
ax.vlines(peak_freq2, noise_height_osc*0.8, peak_height2,
          color="k",
          linestyle="--", lw=1)

# Plot Arrow left and right
ax.annotate("",
            xy=(floor_osc, noise_height_osc*0.8),
            xytext=(peak_freq1, noise_height_osc*0.8),
            arrowprops=dict(arrowstyle="->", color="k", lw=2))
ax.annotate("",
            xy=(floor_osc, noise_height_osc*0.8),
            xytext=(peak_freq2, noise_height_osc*0.8),
            arrowprops=dict(arrowstyle="->", color="k", lw=2))

# Annotate noise floor osc
ax.annotate("",
            xy=(floor_osc, noise_height_osc*0.86),
            xytext=(floor_osc, noise_height_osc*.4),
            arrowprops=dict(arrowstyle="-", color="k", lw=2))
ax.annotate(f"{floor_osc}Hz",
            xy=(floor_osc, noise_height_osc*0.7),
            xytext=(floor_osc*1.02, noise_height_osc*.43))


# Annotate noise floor
ax.annotate("",
            xy=(floor, noise_height*.85),
            xytext=(floor, noise_height*.45),
            arrowprops=dict(arrowstyle="-", color="k", lw=2))
ax.annotate(f"{floor}Hz", xy=(floor, noise_height*.9),
            xytext=(floor*1.05, noise_height*.47))

# plt.grid(True, axis="x", which="minor", ls=":", c='w')

# ax.vlines(floor_osc, 0, 1)

xticks.append(peak_freq2)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ylim = ax.get_ylim()
ax.set_ylim([ylim[0]*.7, ylim[1]])
ax.set_xlabel("Frequency in Hz")
ax.set_ylabel(r"Power in $\mu$V/Hz")
ax.legend(loc=0)
plt.tight_layout()
#plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()



# =============================================================================
# # %%
# fig, ax = plt.subplots(1, 1)
# 
# # No oscillations
# freq_osc, amp, width = [25, 50, 90, 360], [100, 50, 50, 50], [8, 10, 30, 50]
# 
# # Make noise
# w_noise = noise_white(samples-2)
# 
# pink2 = osc_signals(samples, 2, freq_osc, amp, width)
# pink_white2 = pink2 + .13 * w_noise
# 
# freq, psd2_noise_osc = sig.welch(pink_white2, fs=srate, nperseg=nperseg)
# 
# # Bandpass filter between 1Hz and 600Hz
# filt = (freq > 0) & (freq <= 600)
# freq, psd2_noise_osc = freq[filt], psd2_noise_osc[filt]
# 
# # Normalize
# psd2_noise_osc /= psd2_noise_osc.max()
# 
# floor = detect_noise_floor(freq, psd2_noise_osc)
# signal = (freq <= floor)
# noise = (freq >= floor)
# ax.loglog(freq[signal], psd2_noise_osc[signal], "k", label="Subj. 12 STN-L23 Off")
# ax.loglog(freq[noise], psd2_noise_osc[noise], "darkgray")
# 
# floor = detect_noise_floor(freq, psd_lfp_osc)
# signal = (freq <= floor)
# noise = (freq >= floor)
# ax.loglog(freq[signal], psd_lfp_osc[signal], "k", label="Subj. 9 STN-R01 Off")
# ax.loglog(freq[noise], psd_lfp_osc[noise], "darkgray", label="Noise floor")
# 
# plt.show()
# 
# 
# =============================================================================



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
