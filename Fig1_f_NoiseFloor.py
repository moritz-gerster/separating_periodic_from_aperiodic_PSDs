"""Figure 1."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import mne
from pathlib import Path
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from mne.time_frequency import psd_welch
import matplotlib as mpl
import scipy as sp
from numpy.fft import irfft, rfftfreq
import matplotlib.ticker as ticker


def osc_signals(slope=1, periodic_params=None, nlv=None,
                normalize=False, highpass=4, srate=2400,
                duration=180, seed=1):
    """
    Generate colored noise with optionally added oscillations.

    Parameters
    ----------
    slope : float, optional
        Aperiodic 1/f exponent. The default is 1.
    periodic_params : list of tuples, optional
        Oscillations parameters as list of tuples in form
        [(frequency, amplitude, width), (frequency, amplitude, width)].
        The default is None.
    nlv : float, optional
        Level of white noise. The default is None.
    normalize : float, optional
        Normalization factor. The default is 6.
    highpass : int, optional
        The order of the butterworth highpass filter. The default is 4. If None
        no filter will be applied.
    srate : float, optional
        Sample rate of the signal. The default is 2400.
    duration : float, optional
        Duration of the signal in seconds. The default is 180.
    seed : int, optional
        Seed for reproducability. The default is 1.

    Returns
    -------
    noise : ndarray
        Colored noise without oscillations.
    noise_osc : ndarray
        Colored noise with oscillations.
    """
    if seed:
        np.random.seed(seed)
    # Initialize
    n_samples = int(duration * srate)
    amps = np.ones(n_samples//2, complex)
    freqs = rfftfreq(n_samples, d=1/srate)
    freqs = freqs[1:]  # avoid divison by 0

    # Create random phases
    rand_dist = np.random.uniform(0, 2*np.pi, size=amps.shape)
    rand_phases = np.exp(1j * rand_dist)

    # Multiply phases to amplitudes and create power law
    amps *= rand_phases
    amps /= freqs ** (slope / 2)

    # Add oscillations
    amps_osc = amps.copy()
    if periodic_params:
        for osc_params in periodic_params:
            freq_osc, amp_osc, width = osc_params
            amp_dist = sp.stats.norm(freq_osc, width).pdf(freqs)
            # add same random phases
            amp_dist = amp_dist * rand_phases
            amps_osc += amp_osc * amp_dist

    # Create colored noise time series from amplitudes
    noise = irfft(amps)
    noise_osc = irfft(amps_osc)

    # Normalize
    if normalize:
        noise_SD = noise.std()
        scaling = noise_SD / normalize
        noise /= scaling
        noise_osc /= scaling

    # Add white noise
    if nlv:
        w_noise = np.random.normal(scale=nlv, size=n_samples-2)
        noise += w_noise
        noise_osc += w_noise

    # Highpass filter
    if highpass:
        sos = sig.butter(highpass, 1, btype="hp", fs=srate, output='sos')
        noise = sig.sosfilt(sos, noise)
        noise_osc = sig.sosfilt(sos, noise_osc)

    return noise, noise_osc


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


# %% PARAMETERS

# Colors
c_real = "purple"
c_real3 = "mediumvioletred"
c_real2 = "indigo"

c_dummy = "sienna"

c_sim = "k"
c_fit = "b"
c_noise = "darkgray"
# c_right = "k--"

# d)

# c_low = "deepskyblue"
c_med = "limegreen"
c_high = "orangered"

# c_ground = "grey"
# %% Calc a)
srate_emp = 2400
time_emp = np.arange(180 * srate_emp)
samples_emp = time_emp.size
nperseg_emp = srate_emp  # welch

freq_range = [1, 100]

# No oscillations
freq_osc, amp = [23, 360], [0, 0]
slope_a = 2

# Make noise
pink2, _ = osc_signals(slope=2, nlv=0.00005, highpass=False, seed=3)

freq, psd2_noise = sig.welch(pink2, fs=srate_emp, nperseg=nperseg_emp)

# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq, psd2_noise = freq[filt], psd2_noise[filt]

# Normalize
psd2_noise /= psd2_noise.max()

# Detect Noise floor a)
floor_a = detect_noise_floor(freq, psd2_noise)
signal_a = (freq <= floor_a)
noise_a = (freq >= floor_a)


# %% Calc b)

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
                "n_fft": srate_emp,
                "n_overlap": srate_emp // 2,
                "average": "mean"}

spec12, freq = psd_welch(sub12, **welch_params)
spec9, freq = psd_welch(sub9, **welch_params)

psd_lfp_12 = spec12[0]
psd_lfp_9 = spec9[0]

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig1_f_NoiseFloor.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# %% Calc d)

dummy_name = "Paper-Dummy_2016-11-23_F8-F4.npy"
sub_name = "DBS_2018-03-02_STN_L24_rest.npy"

sub_lowN = np.load(path + sub_name)
dummy_lowN = np.load(path + dummy_name)

# =============================================================================
# # Apply highpass filter????
# =============================================================================

srate_lowN = 10000
welch_lowN = dict(fs=srate_lowN, nperseg=srate_lowN)

# Calc PSD
f_lowN, psd_sub_lowN = sig.welch(sub_lowN, **welch_lowN)
f_lowN, psd_dummy_lowN = sig.welch(dummy_lowN, **welch_lowN)

# Detect Noise floor b)
floor_12 = detect_noise_floor(freq, psd_lfp_12)
signal_12 = (freq <= floor_12)
noise_12 = (freq >= floor_12)

floor_9 = detect_noise_floor(freq, psd_lfp_9)
signal_9 = (freq <= floor_9)
noise_9 = (freq >= floor_9)

floor_low_start = detect_noise_floor(f_lowN, psd_sub_lowN, f_start=1)
floor_low_end = detect_noise_floor_end(f_lowN, psd_sub_lowN, f_range=100)

# Fit low noise
freq_range = [1, 95]
fm_lowN = FOOOF(peak_width_limits=[1, 100], verbose=False)
fm_lowN_d = FOOOF(peak_width_limits=[1, 100], max_n_peaks=0, verbose=False)

fm_lowN.fit(f_lowN, psd_sub_lowN, freq_range)
fm_lowN_d.fit(f_lowN, psd_dummy_lowN, freq_range)

exp_lowN = fm_lowN.get_params("aperiodic", "exponent")
exp_lowN_d = fm_lowN_d.get_params("aperiodic", "exponent")

ap_fit_lowN = gen_aperiodic(fm_lowN.freqs, fm_lowN.aperiodic_params_)
ap_fit_lowN_d = gen_aperiodic(fm_lowN_d.freqs, fm_lowN_d.aperiodic_params_)

lowN_fit = fm_lowN.freqs, 10**ap_fit_lowN, "--"
lowN_d_fit = fm_lowN_d.freqs, 10**ap_fit_lowN_d, "--"

fm_lowN_d.plot(plt_log=True)

# %% Calc d

# Get real data
select_times = dict(start=int(0.5*srate_emp), stop=int(185*srate_emp))
sub9_dat = sub9.get_data(**select_times)[0]

# Calc PSD real
welch_params = dict(fs=srate_emp, nperseg=nperseg_emp, detrend=False)
freq, psd_9 = sig.welch(sub9_dat, **welch_params)

filt = (freq > 0) & (freq <= 600)
freq = freq[filt]

psd_9 = psd_9[filt]

# %% Calc b)

slope_s = 1
periodic_params_s = [(10.5, 4, 3),
                     (16, 2, 1.5),
                     (23, 15, 5),
                     (40, 10, 15),
                     (360, 20, 70)]

slope_m = 2
periodic_params_m = [(3, 0.3, .6),
                     (5, .4, 1),
                     (9, 3.3, 2.1),
                     (15, 4, 2.5),
                     (23, 15, 5),
                     (42, 12, 15),
                     (360, 25, 70)]

slope_l = 3
periodic_params_l = [(2, 0.07, .01),
                     (4, 1, 1.5),
                     (5, 1, 2),
                     (9.5, 4, 2),
                     (15, 5, 2.5),
                     (23, 19, 5),
                     (42, 20, 19),
                     (360, 35, 70)]

noise_s, osc_s = osc_signals(slope=slope_s,
                             periodic_params=periodic_params_s,
                             nlv=.0002, normalize=False)

noise_m, osc_m = osc_signals(slope=slope_m,
                             periodic_params=periodic_params_m,
                             nlv=.0002, normalize=False)

noise_l, osc_l = osc_signals(slope=slope_l,
                             periodic_params=periodic_params_l,
                             nlv=.00025, normalize=False)


freq, noise_psd_s = sig.welch(noise_s, **welch_params)
freq, osc_psd_s = sig.welch(osc_s, **welch_params)
freq, noise_psd_m = sig.welch(noise_m, **welch_params)
freq, osc_psd_m = sig.welch(osc_m, **welch_params)
freq, noise_psd_l = sig.welch(noise_l, **welch_params)
freq, osc_psd_l = sig.welch(osc_l, **welch_params)

freq = freq[filt]
noise_psd_s = noise_psd_s[filt]
osc_psd_s = osc_psd_s[filt]
noise_psd_m = noise_psd_m[filt]
osc_psd_m = osc_psd_m[filt]
noise_psd_l = noise_psd_l[filt]
osc_psd_l = osc_psd_l[filt]

# Normalize
norm_s = noise_psd_s[0] / psd_9[0]
norm_m = noise_psd_m[0] / psd_9[0]
norm_l = noise_psd_l[0] / psd_9[0]
noise_psd_s /= norm_s
osc_psd_s /= norm_s
noise_psd_m /= norm_m
osc_psd_m /= norm_m
noise_psd_l /= norm_l
osc_psd_l /= norm_l

# Detect Noise floor b)
floor_s = detect_noise_floor(freq, noise_psd_s)
signal_s = (freq <= floor_s)
noise_s = (freq >= floor_s)

# Fit fooof
fit_params = dict(peak_width_limits=[1, 100], verbose=False)

freq_range = [1, 95]

fm_s = FOOOF(**fit_params)
fm_m = FOOOF(**fit_params)
fm_l = FOOOF(**fit_params)
fm_real = FOOOF(**fit_params)

fm_s.fit(freq, osc_psd_s, freq_range)
fm_m.fit(freq, osc_psd_m, freq_range)
fm_l.fit(freq, osc_psd_l, freq_range)
fm_real.fit(freq, psd_9, freq_range)

exp_s = fm_s.get_params("aperiodic", "exponent")
exp_m = fm_m.get_params("aperiodic", "exponent")
exp_l = fm_l.get_params("aperiodic", "exponent")
exp_real = fm_real.get_params('aperiodic_params', 'exponent')
off_real = fm_real.get_params('aperiodic_params', 'offset')

ap_fit_s = gen_aperiodic(fm_s.freqs, fm_s.aperiodic_params_)
ap_fit_m = gen_aperiodic(fm_m.freqs, fm_m.aperiodic_params_)
ap_fit_l = gen_aperiodic(fm_l.freqs, fm_l.aperiodic_params_)
ap_fit_real = gen_aperiodic(fm_real.freqs, fm_real.aperiodic_params_)

small_fit = fm_s.freqs, 10**ap_fit_s, "--"
med_fit = fm_m.freqs, 10**ap_fit_m, "--"
large_fit = fm_l.freqs, 10**ap_fit_l, "--"
real_fit = fm_real.freqs, 10**ap_fit_real, "--"


# %% Plot


fontsize = 10
# science recommendation:
width = 7.25  # inches
panel_fontsize = 12
legend_fontsize = 9
label_fontsize = 10
tick_fontsize = 9
annotation_fontsize = tick_fontsize

mpl.rcParams['xtick.labelsize'] = tick_fontsize
mpl.rcParams['ytick.labelsize'] = tick_fontsize
mpl.rcParams['axes.labelsize'] = label_fontsize
mpl.rcParams['legend.fontsize'] = legend_fontsize
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


abc = dict(x=0, y=1.01, fontsize=panel_fontsize,
           fontdict=dict(fontweight="bold"))

upper_limits = [10, 50, 100, 200]
alphas = [1, 0.7, .5, .3]

ap_fit_ground = gen_aperiodic(freq, np.array([0, slope_a]))


real_kwargs = dict(c=c_real, alpha=1, lw=2)
small_kwargs = dict(c=c_high, alpha=1, lw=2)
med_kwargs = dict(c=c_med, alpha=1, lw=2)


fill_dic = dict(alpha=0.5)
xticks = [1, 10, 100, 600]
tick_dic = dict(xticks=xticks, xticklabels=xticks, yticks=[])


fig, axes = plt.subplots(2, 2, figsize=[8, width])

# a)
ax = axes[0, 0]
ax.loglog(freq[signal_a], psd2_noise[signal_a], c_sim,
          label=f"1/f a={slope_a} + noise")
ax.loglog(freq, 10**ap_fit_ground, ":", c=c_sim, lw=1,
          label="Ground truth")
xlim = 1, 600
offset = psd2_noise[freq == xlim[1]][0]
ylim = .4 * offset, 1

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
ax.loglog(freq[noise_a], psd2_noise[noise_a], c_noise, label="Plateau")


ax.legend()

xticks = [1] + upper_limits + [600]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ax.set_ylabel("PSD [a.u.]")
ax.text(s="a", **abc, transform=ax.transAxes)

# b)
ax = axes[0, 1]


fill_mask = (freq > 1) & (freq <= freq_range[1])
fill_mask_fit = (fm_s.freqs > 1)

ax.loglog(freq, psd_9, **real_kwargs, zorder=6, label="LFP Sub. 9")
ax.loglog(*real_fit, c=c_real, lw=2, zorder=10, label=f"a={exp_real:.2f}")
ax.loglog(freq, psd_9, **real_kwargs)
ax.loglog(*real_fit, **real_kwargs)

ylim = ax.get_ylim()
ax.set_ylim(ylim)

ax.loglog(freq, osc_psd_s, **small_kwargs, zorder=5, label="Sim a=1")
ax.loglog(*small_fit, c=c_high, lw=2, zorder=9, label=f"a={exp_s:.2f}")
ax.loglog(freq, noise_psd_s, small_kwargs["c"], lw=1, zorder=1,
          label="1/f + noise")
ap_fit_ground = gen_aperiodic(freq, np.array([off_real, slope_s]))
ax.loglog(freq, 10**ap_fit_ground, ":", c=small_kwargs["c"], lw=1, zorder=4,
          label=f"a={slope_s:.0f}")


ax.loglog(freq, osc_psd_m, **med_kwargs, zorder=7, label="Sim a=2")
ax.loglog(*med_fit, c=c_med, lw=2, zorder=8, label=f"a={exp_m:.2f}")
ax.loglog(freq, noise_psd_m, med_kwargs["c"], zorder=2, lw=1,
          label="1/f + noise")
ap_fit_ground = gen_aperiodic(freq, np.array([off_real, slope_m]))
ax.loglog(freq, 10**ap_fit_ground, ":", c=med_kwargs["c"], lw=1, zorder=3,
          label=f"a={slope_m:.0f}")


ax.legend(labelspacing=0.3)
xticks = [1, 10, 100, 600]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ymin, _ = ax.get_ylim()
_, xmax = ax.get_xlim()
ax.set_ylim([ymin, 2])
ax.set_xlim([1, xmax])
ax.text(s="b", **abc, transform=ax.transAxes)

# c)
ax = axes[1, 0]

# Plot Sub 9
ax.loglog(freq[signal_9], psd_lfp_9[signal_9], c_real,
          label="LFP Sub. 9")  # Off STN-R01

# Plot Sub 12
ax.loglog(freq[signal_12], psd_lfp_12[signal_12], c=c_real2,
          label="LFP Sub. 12")  # Off STN-L23

ax.loglog(freq[noise_9], psd_lfp_9[noise_9], c_noise, label="Plateau")
ax.loglog(freq[noise_12], psd_lfp_12[noise_12], c_noise)

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
            xytext=(floor_9*1.02, noise_height_osc*.43),
            fontsize=annotation_fontsize)


# Annotate noise floor
ax.annotate("",
            xy=(floor_12, noise_height*.85),
            xytext=(floor_12, noise_height*.45),
            arrowprops=dict(arrowstyle="-", color=c_sim, lw=2))
ax.annotate(f"{floor_12}Hz", xy=(floor_12, noise_height*.9),
            xytext=(floor_12*1.05, noise_height*.485),
            fontsize=annotation_fontsize)


ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
_, xmax = ax.get_xlim()
ax.set_xlim([1, xmax])
ylim = ax.get_ylim()
ax.set_ylim([ylim[0]*.7, ylim[1]*1.15])
ax.set_xlabel("Frequency in Hz")
ax.set_ylabel(r"PSD [$\mu$$V^2$/Hz]")
ax.legend(loc=0)
ax.text(s="c", **abc, transform=ax.transAxes)


# d)
ax = axes[1, 1]

lowN_signalL = (f_lowN <= floor_low_start)
lowN_signalH = (f_lowN >= floor_low_end)
lowN_plateau = (f_lowN >= floor_low_start) & (f_lowN <= floor_low_end)

ax.loglog(f_lowN[lowN_signalL], psd_sub_lowN[lowN_signalL], c_real3,
          label="LFP low-noise")
ax.loglog(*lowN_fit, c=c_real3, label=f"a={exp_lowN:.2f}")
ax.loglog(f_lowN[lowN_signalH], psd_sub_lowN[lowN_signalH], c_real3)
ax.loglog(f_lowN[lowN_plateau], psd_sub_lowN[lowN_plateau], c_noise)
#          label="Plateau low-noise")

ax.loglog(f_lowN, psd_dummy_lowN, c_dummy, label="Dummy low-noise")
ax.loglog(*lowN_d_fit, c=c_dummy, label=f"a={exp_lowN_d:.2f}")
ax.loglog(freq, psd_lfp_9, c_real, alpha=0.2, label="LFP Sub. 9")

# Plot Plateau lines
ax.vlines(floor_low_start, 0, psd_sub_lowN[f_lowN == floor_low_start],
          color=c_sim,
          linestyle="--", lw=1)
ax.vlines(floor_low_end, 0, psd_sub_lowN[f_lowN == floor_low_end],
          color=c_sim,
          linestyle="--", lw=1)


ax.set_xlabel("Frequency in Hz")
ax.legend()
ax.text(s="d", **abc, transform=ax.transAxes)

xticks = [1, 10, 100, 1000, 5000]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)

locmaj = ticker.LogLocator(base=10, numticks=8)
ax.yaxis.set_major_locator(locmaj)
yticks = ["", "", r"$10^{-5}$", "", r"$10^{-3}$", "", r"$10^{-1}$",
          "", r"$10^1$", "", ""]
ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax.set_yticklabels(yticks)
_, xmax = ax.get_xlim()
ax.set_xlim([1, xmax])
plt.tight_layout()
plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()
