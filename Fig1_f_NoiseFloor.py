"""Figure 1."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import mne
from pathlib import Path
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from mne.time_frequency import psd_welch
from scipy.stats import norm
import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


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


def detect_noise_floor(freq, psd, f_start=1, f_range=50, thresh=0.05,
                       ff_kwargs=dict(verbose=False, max_n_peaks=1)):
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
    f_start -= 1
    exp = 1
    while exp > thresh:
        f_start += 1
        fm = FOOOF(**ff_kwargs)
        fm.fit(freq, psd, [f_start, f_start + f_range])
        exp = fm.get_params('aperiodic_params', 'exponent')
    return f_start + f_range // 2


def detect_noise_floor_end(freq, psd, f_start=None, f_range=50, thresh=0.05,
                           step=None,
                           ff_kwargs=dict(verbose=False, max_n_peaks=1)):
    """
    Detect the end of the noise floor.

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
    if f_start is None:
        f_start = freq[-1]
    if step is None:
        step = f_range / 10
    exp = 1
    while exp > thresh:
        f_start -= step
        fm = FOOOF(**ff_kwargs)
        fm.fit(freq, psd, [f_start - f_range, f_start])
        exp = fm.get_params('aperiodic_params', 'exponent')
    return f_start + f_range // 2


def osc_signals_new(samples, slopes, freq_osc=[], amp=[], width=[],
                    srate=2400, random_phases=False):
    """Return osc signals New."""
    if isinstance(slopes, (float, int)):
        slopes = [slopes]
    # Initialize output
    noises = np.zeros([len(slopes), samples])
    noises_pure = np.zeros([len(slopes), samples])
    # Make fourier amplitudes
    amps = np.ones(samples//2 + 1, complex)
    freqs = np.fft.rfftfreq(samples, d=1/srate)

    # Make 1/f
    freqs[0] = 1  # avoid divison by 0

    for j, slope in enumerate(slopes):
        # Multiply Amp Spectrum by 1/f
        # half slope needed:
        # 1/f^2 in power spectrum = sqrt(1/f^2)=1/f^2*0.5=1/f
        # in amp spectrum
        amps = amps / freqs ** (slope / 2)
        # random phases more realistic, but negative intereference effects
        if random_phases:
            rand_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)
            amps *= np.exp(1j * rand_phases)
        else:
            phase = 0
            amps *= np.exp(1j * phase)
        noises_pure[j] = np.fft.irfft(amps)
        for i in range(len(freq_osc)):
            # make Gaussian peak
            amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
            # normalize peak for smaller amplitude differences
            # for different frequencies
            amp_dist /= np.max(amp_dist)

            amps += amp[i] * amp_dist
    noises[j] = np.fft.irfft(amps)
    return noises, noises_pure  # delete noise_pure


# %% PARAMETERS

# Colors
c_real = "purple"
c_sim = "k"
c_fit = "b"
c_noise = "darkgray"
c_right = "k--"
c_pure = "grey"

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

psd_lfp_12 = spec12[0]
psd_lfp_9 = spec9[0]

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig1_f_NoiseFloor.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)


# d)
dummy_name = "Paper-Dummy_2016-11-23_F8-F4.npy"
sub_name = "DBS_2018-03-02_STN_L24_rest.npy"

sub_low_n = np.load(path + sub_name)
dummy_low_n = np.load(path + dummy_name)

srate_low_n = 10000
welch_low_n = dict(fs=srate_low_n, nperseg=srate_low_n)

# Calc PSD
f_low_n, psd_sub_low_n = sig.welch(sub_low_n, **welch_low_n)
f_low_n, psd_dummy_low_n = sig.welch(dummy_low_n, **welch_low_n)


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

# Detect Noise floor a)
floor_a = detect_noise_floor(freq, psd2_noise)
signal_a = (freq <= floor_a)
noise_a = (freq >= floor_a)

# Detect Noise floor b)
floor_12 = detect_noise_floor(freq, psd_lfp_12)
signal_12 = (freq <= floor_12)
noise_12 = (freq >= floor_12)

floor_9 = detect_noise_floor(freq, psd_lfp_9)
signal_9 = (freq <= floor_9)
noise_9 = (freq >= floor_9)

# Detect Noise floor b)
floor_low_start = detect_noise_floor(f_low_n, psd_sub_low_n, f_start=1)
floor_low_end = detect_noise_floor_end(f_low_n, psd_sub_low_n, f_range=100)

# %% Plot without knee
abc = dict(x=0, y=1.01, fontsize=14, fontdict=dict(fontweight="bold"))

upper_limits = [10, 50, 100, 200]
alphas = [1, 0.7, .5, .3]

ap_fit_ground = gen_aperiodic(freq, np.array([0, slope]))


fig, axes = plt.subplots(2, 2, figsize=[9, 8],
                         gridspec_kw=dict(width_ratios=[.9, 1]))

ax = axes[0, 0]
ax.loglog(freq[signal_a], psd2_noise[signal_a], c_sim, label="PSD")
ax.loglog(freq[noise_a], psd2_noise[noise_a], c_noise, label="Plateau")
ax.loglog(freq, 10**ap_fit_ground, ":", c=c_sim, lw=1,
          label=f"Ground truth a={slope}")

xlim = 1, 600
offset = psd2_noise[freq == xlim[1]][0]
ylim = .5 * offset, 1

ax.set_ylim(ylim)
ax.set_xlim(xlim)

for i, lim in enumerate(upper_limits):
    fm = FOOOF(verbose=False)
    fm.fit(freq, psd2_noise, [1, lim])
    exp = fm.get_params('aperiodic_params', 'exponent')
    ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
    ax.loglog(fm.freqs, 10**ap_fit, "-", c=c_fit, lw=2, alpha=alphas[i]-0.1,
              label=f"1-{lim}Hz a={exp:.2f}")  # c="dodgerblue"
    # annotate x-crossing
    ax.vlines(lim, ylim[0], 10**ap_fit[-1], color=c_sim, linestyle="-", lw=.3)

ax.legend()

xticks = [1] + upper_limits + [600]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ax.set_xlabel("Frequency in Hz")
ax.set_ylabel("PSD [a.u.]")
ax.text(s="a", **abc, transform=ax.transAxes)


ax = axes[0, 1]

# Plot Sub 12
ax.loglog(freq[signal_12], psd_lfp_12[signal_12], c_real,
          label="LFP Sub. 12")  # Off STN-L23
ax.loglog(freq[noise_12], psd_lfp_12[noise_12], c_noise)

# Plot Sub 9
ax.loglog(freq[signal_9], psd_lfp_9[signal_9], c_real,
          label="LFP Sub. 9")  # Off STN-R01
ax.loglog(freq[noise_9], psd_lfp_9[noise_9], c_noise, label="Plateau")

# Get peak freqs, heights, and noise heights
peak_freq1 = 23
peak_freq2 = 350

peak_height1 = psd_lfp_9[freq == peak_freq1]
peak_height2 = psd_lfp_9[freq == peak_freq2]

noise_height = psd_lfp_12[freq == floor_12]
noise_height_osc = psd_lfp_9[freq == floor_9]

# Plot Peak lines
ax.vlines(peak_freq1, noise_height_osc*0.8, peak_height1,
          color=c_sim,
          linestyle="--", lw=1)
ax.vlines(peak_freq2, noise_height_osc*0.8, peak_height2,
          color=c_sim,
          linestyle="--", lw=1)

# Plot Arrow left and right
ax.annotate("",
            xy=(floor_9, noise_height_osc*0.8),
            xytext=(peak_freq1, noise_height_osc*0.8),
            arrowprops=dict(arrowstyle="->", color=c_sim, lw=2))
ax.annotate("",
            xy=(floor_9, noise_height_osc*0.8),
            xytext=(peak_freq2, noise_height_osc*0.8),
            arrowprops=dict(arrowstyle="->", color=c_sim, lw=2))

# Annotate noise floor osc
ax.annotate("",
            xy=(floor_9, noise_height_osc*0.86),
            xytext=(floor_9, noise_height_osc*.4),
            arrowprops=dict(arrowstyle="-", color=c_sim, lw=2))
ax.annotate(f"{floor_9}Hz",
            xy=(floor_9, noise_height_osc*0.7),
            xytext=(floor_9*1.02, noise_height_osc*.43))


# Annotate noise floor
ax.annotate("",
            xy=(floor_12, noise_height*.85),
            xytext=(floor_12, noise_height*.45),
            arrowprops=dict(arrowstyle="-", color=c_sim, lw=2))
ax.annotate(f"{floor_12}Hz", xy=(floor_12, noise_height*.9),
            xytext=(floor_12*1.05, noise_height*.485))

# plt.grid(True, axis="x", which="minor", ls=":", c='w')
# ax.vlines(floor_osc, 0, 1)

# xticks.append(peak_freq2)
xticks = [1, 10, 100, 600]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ylim = ax.get_ylim()
ax.set_ylim([ylim[0]*.7, ylim[1]])
ax.set_xlabel("Frequency in Hz")
ax.set_ylabel(r"PSD [$\mu$$V^2$/Hz]")
ax.legend(loc=0)
ax.text(s="b", **abc, transform=ax.transAxes)


# d)
ax = axes[1, 1]
ax.loglog(f_low_n, psd_sub_low_n, c_real, label="low-noise LFP")
ax.loglog(f_low_n, psd_dummy_low_n, c_noise, label="low-noise empty room")

# Plot Plateau lines
ax.vlines(floor_low_start, 0, psd_sub_low_n[f_low_n == floor_low_start],
          color=c_sim,
          linestyle="--", lw=1)
ax.vlines(floor_low_end, 0, psd_sub_low_n[f_low_n == floor_low_end],
          color=c_sim,
          linestyle="--", lw=1)

xticks = [1, 10, 100, 1000, 5000]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ax.set_xlabel("Frequency in Hz")
ax.set_ylabel(r"PSD [$\mu$$V^2$/Hz]")
ax.legend()
ax.text(s="d", **abc, transform=ax.transAxes)

plt.tight_layout()
plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()

# %% Calc C

# Oscillatory parameters
# =============================================================================
# freq_osc_s = [2.5,   3, 4, 7,  27,  36,  360]
# amp_s =      [1.5, 4.5, 5, 4, 750, 500, 7000]
# width_s =    [0.1, .7, 1.2, 20, 7, 11, 60]
# slope_s = 0.75
# =============================================================================

# =============================================================================
# # Works:
# freq_osc_m = [ 1, 6,   8,  23,  55, 360]
# amp_m =      np.array([ 3, 4,  19, 300, 150, 2000]) * 1
# width_m =    [.8, 2, 1.9,   6,  10, 50]
# slope_m = 1
# =============================================================================

# =============================================================================
# freq_osc_l = [2.5, 3, 5, 27, 36, 360]
# amp_l =      [2, 4, 6, 670, 480, 7000]
# width_l =    [1.2, 1.2, 1.4, 7, 11, 60]
# slope_l = 1.25
# =============================================================================

# =============================================================================
# freq_osc3 = [2.5, 3, 5, 27, 36, 360]
# amp3 =      [160, 1000, 400, 67000, 48000, 700000]
# width3 =    [0.2, 1.2, 1.4, 7, 11, 60]
# slope3 = 3
# =============================================================================

# # Works:
freq_osc_m = [ 1, 6,   8,  23,  55, 360]
amp_m =      [ 0, 0,  0, 0, 0, 0]
width_m =    [.8, 2, 1.9,   6,  10, 50]
slope_m = .8

# Gen signal
oscs_s = slope_s, freq_osc_s, amp_s, width_s
oscs_m = slope_m, freq_osc_m, amp_m, width_m
oscs_l = slope_l, freq_osc_l, amp_l, width_l
# oscs3 = slope3, freq_osc3, amp3, width3

pink_s, pure_s = osc_signals_new(samples, *oscs_s)
pink_m, pure_m = osc_signals_new(samples, *oscs_m)
pink_l, pure_l = osc_signals_new(samples, *oscs_l)
# pink3, pure3 = osc_signals_new(samples, *oscs3)

# Highpass filter
sos = sig.butter(20, 1, btype="hp", fs=srate, output='sos')
pink_m = sig.sosfilt(sos, pink_m)
pure_m = sig.sosfilt(sos, pure_m)

# Normalize amplitudes
sub9_dat = sub9.get_data()
#sub9_norm = (sub9_dat - sub9_dat.mean()) / sub9_dat.std()
freq, psd_9_norm = sig.welch(sub9_norm[0][int(0.5*srate):int(185*srate)], fs=srate, nperseg=nperseg,
                             detrend="linear")

spec9, freq = psd_welch(sub9, **welch_params)
psd_lfp_9 = spec9[0]

pink_m = (pink_m - pink_m.mean()) / pink_m.std()
pure_m = (pure_m - pure_m.mean()) / pure_m.std()



pink_s = pink_s[0]
pure_s = pure_s[0]
pink_m = pink_m[0]
pure_m = pure_m[0]
pink_l = pink_l[0]
pure_l = pure_l[0]
# pink3 = pink3[0]
# pure3 = pure3[0]

# Add white noise
w_noise = noise_white(samples)
# pink_s += .00035 * w_noise
# pure_s += .00035 * w_noise
pink_m += .05 * w_noise
#pure_m += .00022 * w_noise
pure_m += .05 * w_noise
# pink_l += .0007 * w_noise
# pure_l += .0007 * w_noise
# pink3 += .037 * w_noise
# pure3 += .037 * w_noise

# Calc PSD
freq, sim_s = sig.welch(pink_s, fs=srate, nperseg=nperseg, detrend=False)
freq, sim_m = sig.welch(pink_m, fs=srate, nperseg=nperseg, detrend=False)
freq, sim_l = sig.welch(pink_l, fs=srate, nperseg=nperseg, detrend=False)
# freq, sim3 = sig.welch(pink3, fs=srate, nperseg=nperseg, detrend=False)

freq, pure_s = sig.welch(pure_s, fs=srate, nperseg=nperseg, detrend=False)
freq, pure_m = sig.welch(pure_m, fs=srate, nperseg=nperseg, detrend=False)
freq, pure_l = sig.welch(pure_l, fs=srate, nperseg=nperseg, detrend=False)
# freq, pure3 = sig.welch(pure3, fs=srate, nperseg=nperseg, detrend=False)

# Mask between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]

psd_9_norm = psd_9_norm[filt]

sim_s = sim_s[filt]
sim_m = sim_m[filt]
sim_l = sim_l[filt]
# sim3 = sim3[filt]

pure_s = pure_s[filt]
pure_m = pure_m[filt]
pure_l = pure_l[filt]
# pure3 = pure3[filt]

# Adjust offset for real spectrum
psd_9_norm /= psd_9_norm[-1]
sim_s /= sim_s[0]
sim_m /= sim_m[-1]
sim_l /= sim_l[0]
# sim3 /= sim3[0]

pure_s /= pure_s[0]
pure_m /= pure_m[-1]
pure_l /= pure_l[0]
# pure3 /= pure3[0]

# % C: Plot
fig, axes = plt.subplots(1, 1, figsize=[8, 8])
ax = axes
ax.loglog(freq, psd_9_norm, c_real, alpha=0.4, label="LFP Sub. 9")
ax.loglog(freq, psd_lfp_9, c_real, alpha=0.4, label="LFP Sub. 9")
# =============================================================================
# ax.loglog(freq, sim_m, c_sim, label=f"Sim a={slope_s}")
ax.loglog(freq, sim_m, c_sim, label=f"Sim a={slope_m}")
# ax.loglog(freq, sim_l, ":", c=c_sim, , label=f"Sim a={slope_l}")
# ax.loglog(freq, sim3, "--", c=c_sim, label="Sim a=3")
# =============================================================================
# =============================================================================
# ax.loglog(freq, pure_s, c_noise, label=f"1/f a={slope_s}")
ax.loglog(freq, pure_m, c_noise, label=f"1/f a={slope_m}")
# ax.loglog(freq, pure_l, ":", c=c_noise, label=f"1/f a={slope_l}")
# =============================================================================
ax.set_title(f"Freqs: {freq_osc_m}\nAmps: {amp_m}\nWidths: {width_m}")
ax.legend()
plt.show()

# %%

fm = FOOOF()
fm.fit(freq, psd_9_norm, [1, 600])
fm.report()
# %% Old
# =============================================================================
# # %% Move noise floor
#
# # oscillations
# freq_osc1 = [2.5, 3, 4, 7, 22, 360]
# amp1 =      [1.5, 4.5, 5, 3, 600, 6000]
# width1 =    [0.1, .7, 1.2, 20, 6, 60]
#
# freq_osc2 = [2.5, 3, 4, 7, 27, 36, 360]
# amp2 =      [1.5, 4.5, 5, 3, 750, 500, 6000]
# width2 =    [0.1, .7, 1.2, 20, 7, 11, 60]
# slopes = [1]
#
# freq_osc3 = [2.5, 3, 4, 7, 35, 80]
# amp3 =      [1.5, 4.5, 5, 3, 2000, 2000]
# width3 =    [0.1, .7, 1.2, 20, 11, 18]
#
# # oscs1 = freq_osc1, amp1, width1
# oscs2 = freq_osc2, amp2, width2
# oscs3 = freq_osc3, amp3, width3
#
# # Make noise
# w_noise = noise_white(samples)
#
# # pink_left, _ = osc_signals_new(samples, slopes, *oscs1)
# pink_middle, pure_middle = osc_signals_new(samples, slopes, *oscs2)
# pink_right, _ = osc_signals_new(samples, slopes, *oscs3)
#
# # pink_left = pink_left[0]
# pink_middle = pink_middle[0]
# pure_middle = pure_middle[0]
# pink_right = pink_right[0]
#
# # pink_left += .0005 * w_noise
# pink_middle += .0005 * w_noise
# pure_middle += .0005 * w_noise
# pink_right += .0005 * w_noise
#
# welch = dict(fs=srate, nperseg=nperseg, detrend=False)
# # freq, sim_left = sig.welch(pink_left, **welch)
# freq, sim_middle = sig.welch(pink_middle, **welch)
# freq, pure_middle = sig.welch(pure_middle, **welch)
# freq, sim_right = sig.welch(pink_right, **welch)
#
# # Bandpass filter between 1Hz and 600Hz
# filt = (freq > 0) & (freq <= 600)
#
# freq = freq[filt]
# # sim_left = sim_left[filt]
# sim_middle = sim_middle[filt]
# pure_middle = pure_middle[filt]
# sim_right = sim_right[filt]
#
# # Adjust offset for real spectrum
# # sim_left /= sim_left[-1]
# sim_middle /= sim_middle[-1]
# pure_middle /= pure_middle[-1]
# sim_right /= sim_right[-1]
#
# find_noise = dict(f_start=30, f_range=30, thresh=0,
#                   ff_kwargs=dict(max_n_peaks=0, verbose=False))
#
# # nfloor_left = detect_noise_floor(freq, sim_left, **find_noise)
# nfloor_pure = detect_noise_floor(freq, pure_middle, **find_noise)
# nfloor_middle = detect_noise_floor(freq, sim_middle, **find_noise)
# nfloor_right = detect_noise_floor(freq, sim_right, **find_noise)
#
# # %% Plot
#
# fig, axes = plt.subplots(1, 1, figsize=[8, 8])
# ax = axes
#
# ax.loglog(freq, spec10_on, c_real, alpha=0.4, label=ch + " on",)
#
# ax.loglog(freq, sim_middle, c_sim, label="Sim2")
#
# ax.loglog(freq, sim_right, c_right, label="Sim3")
#
# ax.loglog(freq, pure_middle, c_pure, label=f"Ground truth a={slopes[0]}")
#
# # annotate noise floors
# ax.vlines(nfloor_middle, 1, 0, color=c_sim)
# ax.vlines(nfloor_right, 1, 0, color=c_sim, ls="--")
#
# ax.legend()
# plt.show()
#
# =============================================================================
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
# ax.loglog(freq[signal], psd2_noise_osc[signal], c_sim,
#            label="Subj. 12 STN-L23 Off")
# ax.loglog(freq[noise], psd2_noise_osc[noise], "darkgray")
#
# floor = detect_noise_floor(freq, psd_lfp_9)
# signal = (freq <= floor)
# noise = (freq >= floor)
# ax.loglog(freq[signal], psd_lfp_9[signal], c_sim, label="Subj.
# 9 STN-R01 Off")
# ax.loglog(freq[noise], psd_lfp_9[noise], "darkgray", label="Noise floor")
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
# ax.loglog(freq, psd2_noise, c_sim, label="Spectrum", lw=3)
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
