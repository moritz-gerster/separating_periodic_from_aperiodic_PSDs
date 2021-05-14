"""Figure 1."""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig
import mne
from pathlib import Path
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic


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
        [(frequency, amplitude, width), (frequency, amplitude, width)] for
        two oscillations.
        The default is None.
    nlv : float, optional
        Level of white noise. The default is None.
    normalize : float, optional
        Normalization factor. The default is 6.
    highpass : int, optional
        The order of the butterworth highpass filter. The default is 4. If
        None, no filter will be applied.
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

# =============================================================================
#     # Normalize
#     if normalize:
#         noise_SD = noise.std()
#         scaling = noise_SD / normalize
#         noise /= scaling
#         noise_osc /= scaling
# =============================================================================

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


def detect_noise_floor(freq, psd, f_start, f_range=50, thresh=0.05,
                       step=1, reverse=False,
                       ff_kwargs=dict(verbose=False, max_n_peaks=1)):
    """
    Detect the plateau of a power spectrum where the slope a < thresh.

    Parameters
    ----------
    freq : ndarray
        Freq array.
    psd : ndarray
        PSD array.
    f_start : float
        Starting frequency for the search.
    f_range : int, optional
        Fitting range.
        If set low, more susceptibility to noise/peaks.
        If set large, less spatial precision.
        The default is 50.
    thresh : float, optional
        Threshold for plateau. The default is 0.05.
    step : int, optional
        Step of loop over fitting range. The default is 1 which might take
        unneccessarily long computation time for maximum precision.
    reverse : bool, optional
        If True, start at high frequencies and detect the end of a pleateau.
        The default is False.
    ff_kwargs : dict, optional
        Fooof fitting keywords.
        The default is dict(verbose=False, max_n_peaks=1). There shouldn't be
        peaks close to the plateau but fitting at least one peak is a good
        idea for power line noise.

    Returns
    -------
    n_start : int
        Start frequency of plateau.
        If reverse=True, end frequency of plateau.
    """
    exp = 1
    fm = FOOOF(**ff_kwargs)
    while exp > thresh:
        if reverse:
            f_start -= step
            freq_range = [f_start - f_range, f_start]
        else:
            f_start += step
            freq_range = [f_start, f_start + f_range]
        fm.fit(freq, psd, freq_range)
        exp = fm.get_params('aperiodic_params', 'exponent')
    return f_start + f_range // 2


# %% Parameters

# Colors
# a)
c_sim = "k"
c_fit = "b"  # c="dodgerblue"
c_noise = "darkgray"

# b)
c_real = "purple"
c_low = "limegreen"
c_high = "orangered"

# c)
c_real2 = "indigo"

# d)
c_real3 = "mediumvioletred"
c_dummy = "sienna"

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig1_f_NoiseFloor.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# %% Load data

# Litvak file names
path = "../data/Fig1/"
fname12 = "subj12_off_R7_raw.fif"
fname9 = "subj9_off_R1_raw.fif"

# Low noise file names
dummy_name = "Paper-Dummy_2016-11-23_F8-F4.npy"
sub_lowN = "DBS_2018-03-02_STN_L24_rest.npy"

sub12 = mne.io.read_raw_fif(path + fname12, preload=True)
sub9 = mne.io.read_raw_fif(path + fname9, preload=True)

sub_lowN = np.load(path + sub_lowN)
dummy_lowN = np.load(path + dummy_name)

# Select channels
sub12.pick_channels(['STN_L23'])
sub9.pick_channels(['STN_R01'])

# Filter out line noise
filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}

sub12.notch_filter(**filter_params)
sub9.notch_filter(**filter_params)
# filtering of low noise data not neccessary

# Convert mne to numpy
srate = 2400
srate_lowN = 10000
start = int(0.5 * srate)  # artefact in beginning of recording
stop = int(185 * srate)  # artefact at the end of recording

sub12 = sub12.get_data(start=start, stop=stop)[0]
sub9 = sub9.get_data(start=start, stop=stop)[0]

# Calc Welch
welch_params = dict(fs=srate, nperseg=srate)  # detrend=False?
welch_lowN = dict(fs=srate_lowN, nperseg=srate_lowN)

freq, psd_sub9 = sig.welch(sub9, **welch_params)
freq, psd_sub12 = sig.welch(sub12, **welch_params)

# Mask above highpass and below lowpass
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
psd_sub12 = psd_sub12[filt]
psd_sub9 = psd_sub9[filt]

f_lowN, psd_sub_lowN = sig.welch(sub_lowN, **welch_lowN)
f_lowN, psd_dummy_lowN = sig.welch(dummy_lowN, **welch_lowN)

# Mask close to Nyquist frequency
mask = (f_lowN <= 4900)
f_lowN = f_lowN[mask]
psd_sub_lowN = psd_sub_lowN[mask]
psd_dummy_lowN = psd_dummy_lowN[mask]
# %% Calc a)

# Make noise
slope_a = 2
noise_params_a = dict(nlv=0.00005, highpass=False, seed=3, slope=slope_a)
pink2, _ = osc_signals(**noise_params_a)

# Calc PSD
freq, psd2_noise = sig.welch(pink2, **welch_params)

# Mask above highpass and below lowpass
freq, psd2_noise = freq[filt], psd2_noise[filt]

# Normalize
psd2_noise /= psd2_noise[0]

# Detect Noise floor
floor_a = detect_noise_floor(freq, psd2_noise, f_start=1)
signal_a = (freq <= floor_a)
noise_a = (freq >= floor_a)

ground_truth = gen_aperiodic(freq, np.array([0, slope_a]))

# %% Plot params a)
plot_sim = (freq[signal_a], psd2_noise[signal_a], c_sim)
plot_plateau = (freq[noise_a], psd2_noise[noise_a], c_noise)
plot_ground = (freq, 10**ground_truth, c_sim)

# plot limits, ticks, and labels
xlim_a = (1, 600)
y_plateau = psd2_noise[freq == xlim_a[1]][0]
ylim_a = (.4*y_plateau, 1)
xticks_a = [1, 10, 50, 100, 200, 600]
ylabel_a = "PSD [a.u.]"
axes_a = dict(xlim=xlim_a, ylim=ylim_a, xticks=xticks_a, xticklabels=xticks_a,
              ylabel=ylabel_a)

# fit for different upper fit limits and plot
plot_fits, dic_fits, vlines = [], [], []

upper_fit_limits = xticks_a[1:-1]
fit_alphas = [.9, 0.6, .4, .2]

fm = FOOOF(verbose=False)
for i, lim in enumerate(upper_fit_limits):
    fm.fit(freq, psd2_noise, [1, lim])
    exp = fm.get_params('aperiodic_params', 'exponent')
    fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
    label = f"1-{lim}Hz a={exp:.2f}"
    plot_fit = fm.freqs, 10**fit, "-"
    dic_fit = dict(c=c_fit, lw=2, alpha=fit_alphas[i], label=label)
    # annotate x-crossing
    vline = lim, ylim_a[0], 10**fit[-1]
    plot_fits.append(plot_fit)
    dic_fits.append(dic_fit)
    vlines.append(vline)
dic_line = dict(color=c_sim, linestyle="-", lw=.3)

# %% Calc b)

# Simulate psd_sub9
slope1 = 1
slope2 = 2

osc_params1 = [(10.5, 4, 3),
               (16, 2, 1.5),
               (23, 15, 5),
               (40, 10, 15),
               (360, 20, 70)]

osc_params2 = [(3, 0.3, .6),
               (5, .4, 1),
               (9, 3.3, 2.1),
               (15, 4, 2.5),
               (23, 15, 5),
               (42, 12, 15),
               (360, 25, 70)]

noise1, osc1 = osc_signals(slope=slope1,
                           periodic_params=osc_params1,
                           nlv=.0002, normalize=False)

noise2, osc2 = osc_signals(slope=slope2,
                           periodic_params=osc_params2,
                           nlv=.0002, normalize=False)

# Calc Welch
freq, psd_noise1 = sig.welch(noise1, **welch_params)
freq, psd_noise2 = sig.welch(noise2, **welch_params)
freq, psc_osc1 = sig.welch(osc1, **welch_params)
freq, psc_osc2 = sig.welch(osc2, **welch_params)

# Filter above highpass and below lowpass
freq = freq[filt]
psd_noise1 = psd_noise1[filt]
psd_noise2 = psd_noise2[filt]
psc_osc1 = psc_osc1[filt]
psc_osc2 = psc_osc2[filt]

# Normalize
norm1 = psd_noise1[0] / psd_sub9[0]
norm2 = psd_noise2[0] / psd_sub9[0]
psd_noise1 /= norm1
psc_osc1 /= norm1
psd_noise2 /= norm2
psc_osc2 /= norm2

# Fit fooof
freq_range = [1, 95]  # upper border above oscillations range, below plateau
fooof_params = dict(peak_width_limits=[1, 100], verbose=False)

fm1 = FOOOF(**fooof_params)
fm2 = FOOOF(**fooof_params)
fm_sub9 = FOOOF(**fooof_params)

fm1.fit(freq, psc_osc1, freq_range)
fm2.fit(freq, psc_osc2, freq_range)
fm_sub9.fit(freq, psd_sub9, freq_range)

# Extract fit results
exp1 = fm1.get_params("aperiodic", "exponent")
exp2 = fm2.get_params("aperiodic", "exponent")
exp_sub9 = fm_sub9.get_params('aperiodic_params', 'exponent')
off_sub9 = fm_sub9.get_params('aperiodic_params', 'offset')

# Simulate fitting results
ap_fit1 = gen_aperiodic(fm1.freqs, fm1.aperiodic_params_)
ap_fit2 = gen_aperiodic(fm2.freqs, fm2.aperiodic_params_)
ap_fit_sub9 = gen_aperiodic(fm_sub9.freqs, fm_sub9.aperiodic_params_)

fit1 = fm1.freqs, 10**ap_fit1, c_high
fit2 = fm2.freqs, 10**ap_fit2, c_low
fit_sub9 = fm_sub9.freqs, 10**ap_fit_sub9, c_real

# %% Plot params b)
plot_sub9 = freq, psd_sub9, c_real

plot_osc1 = freq, psc_osc1, c_high
plot_osc2 = freq, psc_osc2, c_low

plot_noise1 = freq, psd_noise1, c_high
plot_noise2 = freq, psd_noise2, c_low

ground1 = gen_aperiodic(freq, np.array([off_sub9, slope1]))
ground2 = gen_aperiodic(freq, np.array([off_sub9, slope2]))

plot_ground1 = freq, 10**ground1, c_high
plot_ground2 = freq, 10**ground2, c_low

xticks_b = [1, 10, 100, 600]
ylim_b = (0.006, 2)
xlim_b = (1, 825)

axes_b = dict(xlim=xlim_b, ylim=ylim_b, xticks=xticks_b, xticklabels=xticks_b)


# %% Plot params c)

# Detect Noise floor
floor_12 = detect_noise_floor(freq, psd_sub12, f_start=1)
floor_9 = detect_noise_floor(freq, psd_sub9, f_start=1)

# Mask signal/noise
signal_12 = (freq <= floor_12)
signal_9 = (freq <= floor_9)

noise_12 = (freq >= floor_12)
noise_9 = (freq >= floor_9)

# Prepare plot
plot_sub9_signal = (freq[signal_9], psd_sub9[signal_9], c_real)
plot_sub12_signal = (freq[signal_12], psd_sub12[signal_12], c_real2)

plot_sub9_plateau = (freq[noise_9], psd_sub9[noise_9], c_noise)
plot_sub12_plateau = (freq[noise_12], psd_sub12[noise_12], c_noise)

# Get Oscillation coordinates sub9
peak_freq1 = 23
peak_freq2 = 350

peak_height1 = psd_sub9[freq == peak_freq1]
peak_height2 = psd_sub9[freq == peak_freq2]

noise_height = psd_sub12[freq == floor_12]
noise_height_osc = psd_sub9[freq == floor_9]

# Create lines, arrows, and text to annotate noise floor
line_osc1 = dict(x=peak_freq1, ymin=noise_height_osc*0.8, ymax=peak_height1,
                 color=c_sim, linestyle="--", lw=1)

line_osc2 = dict(x=peak_freq2, ymin=noise_height_osc*0.8, ymax=peak_height2,
                 color=c_sim, linestyle="--", lw=1)

arrow1 = dict(text="",
              xy=(floor_9, noise_height_osc*0.8),
              xytext=(peak_freq1, noise_height_osc*0.8),
              arrowprops=dict(arrowstyle="->", color=c_sim, lw=2))
arrow2 = dict(text="",
              xy=(floor_9, noise_height_osc*0.8),
              xytext=(peak_freq2, noise_height_osc*0.8),
              arrowprops=dict(arrowstyle="->", color=c_sim, lw=2))

plateau_line9 = dict(text="",
                     xy=(floor_9, noise_height_osc*0.86),
                     xytext=(floor_9, noise_height_osc*.4),
                     arrowprops=dict(arrowstyle="-", color=c_sim, lw=2))

plateau_line12 = dict(text="",
                      xy=(floor_12, noise_height*.85),
                      xytext=(floor_12, noise_height*.45),
                      arrowprops=dict(arrowstyle="-", color=c_sim, lw=2))

plateau_txt9 = dict(text=f"{floor_9}Hz",
                    xy=(floor_9, noise_height_osc*0.7),
                    xytext=(floor_9*1.02, noise_height_osc*.43))

plateau_txt12 = dict(text=f"{floor_12}Hz", xy=(floor_12, noise_height*.9),
                     xytext=(floor_12*1.05, noise_height*.485))

xlim_c = (1, 826)
ylim_c = (0.001, 2)
xlabel_c = "Frequency in Hz"
ylabel_c = r"PSD [$\mu$$V^2$/Hz]"

axes_c = dict(xlabel=xlabel_c, ylabel=ylabel_c, xlim=xlim_c, ylim=ylim_c,
              xticks=xticks_b, xticklabels=xticks_b)
# %% Calc d)

# Detect Noise floor
floor_lowN_start = detect_noise_floor(f_lowN, psd_sub_lowN, f_start=1)
floor_lowN_end = detect_noise_floor(f_lowN, psd_sub_lowN, f_start=f_lowN[-1],
                                    f_range=100, step=10, reverse=True)

# Fit low noise spectra
freq_range = [1, 45]  # no 1/f power law after 45Hz

fm_lowN_sub = FOOOF(**fooof_params)
fm_lowN_dummy = FOOOF(**fooof_params, max_n_peaks=0)  # no peaks in dummy

fm_lowN_sub.fit(f_lowN, psd_sub_lowN, freq_range)
fm_lowN_dummy.fit(f_lowN, psd_dummy_lowN, freq_range)

exp_lowN = fm_lowN_sub.get_params("aperiodic", "exponent")
exp_lowN_d = fm_lowN_dummy.get_params("aperiodic", "exponent")

fit_lowN_sub = gen_aperiodic(fm_lowN_sub.freqs,
                             fm_lowN_sub.aperiodic_params_)
fit_lowN_dummy = gen_aperiodic(fm_lowN_dummy.freqs,
                               fm_lowN_dummy.aperiodic_params_)

# %% Plot params d)
plot_lowN_fit_sub = fm_lowN_sub.freqs, 10**fit_lowN_sub, c_real3
plot_lowN_fit_dummy = fm_lowN_dummy.freqs, 10**fit_lowN_dummy, c_dummy

signal_low = (f_lowN <= floor_lowN_start)
signal_high = (f_lowN >= floor_lowN_end)
signal_plateau = (f_lowN >= floor_lowN_start) & (f_lowN <= floor_lowN_end)

plot_sub_lowN_low = (f_lowN[signal_low], psd_sub_lowN[signal_low], c_real3)
plot_sub_lowN_high = (f_lowN[signal_high], psd_sub_lowN[signal_high], c_real3)
plot_sub_lowN_plateau = (f_lowN[signal_plateau], psd_sub_lowN[signal_plateau],
                         c_noise)

dummy_lowN = (f_lowN, psd_dummy_lowN, c_dummy)

plateu_line_start = dict(x=floor_lowN_start,
                         ymin=0,
                         ymax=psd_sub_lowN[f_lowN == floor_lowN_start],
                         color=c_sim, ls="--", lw=1)
plateu_line_end = dict(x=floor_lowN_end,
                       ymin=0,
                       ymax=psd_sub_lowN[f_lowN == floor_lowN_end],
                       color=c_sim, ls="--", lw=1)


xlabel_d = "Frequency in Hz"
xticks_d = [1, 10, 100, 1000, 5000]
xlim_d = (1, 7490)
yticklabels_d = ["", "", r"$10^{-5}$", "", r"$10^{-3}$", "", r"$10^{-1}$",
                 "", r"$10^1$", "", ""]

axes_d = dict(xlabel=xlabel_d, xticks=xticks_d, xticklabels=xticks_d,
              xlim=xlim_d, yticklabels=yticklabels_d)

# %% Plot Params

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

panel_labels = dict(x=0, y=1.01, fontsize=panel_fontsize,
                    fontdict=dict(fontweight="bold"))
lw_noise = 1
line_fit = dict(lw=2, ls="--")
line_ground = dict(lw=1, ls=":")

# %% Plot
fig, axes = plt.subplots(2, 2, figsize=[8, width])

# a)  =========================================================================
ax = axes[0, 0]

# Plot simulated PSD and ground truth
ax.loglog(*plot_sim, label=f"1/f a={slope_a} + noise")
ax.loglog(*plot_ground, **line_ground, label="Ground truth")

# Plot fits for different upper fitting borders
for i in range(len(upper_fit_limits)):
    ax.loglog(*plot_fits[i], **dic_fits[i])
    ax.vlines(*vlines[i], **dic_line)

# Plot plateau in grey
ax.loglog(*plot_plateau, label="Plateau")

# Set axes
ax.set(**axes_a)
ax.legend()
# =============================================================================

# b)  =========================================================================
ax = axes[0, 1]

# Plot spectra
ax.loglog(*plot_sub9, zorder=6, label="LFP Sub. 9")
ax.loglog(*plot_osc1, zorder=5, label="Sim a=1")
ax.loglog(*plot_osc2, zorder=7, label="Sim a=2")

# Plot fits
ax.loglog(*fit_sub9, **line_fit, zorder=10,
          label=r"$a_{fit}$="f"{exp_sub9:.2f}")
ax.loglog(*fit1, **line_fit, zorder=9, label=r"$a_{fit}$="f"{exp1:.2f}")
ax.loglog(*fit2, **line_fit, zorder=8, label=r"$a_{fit}$="f"{exp2:.2f}")

# Plot ground truth sim1
# ax.loglog(*plot_ground1, **line_ground, zorder=4, label=f"a={slope1:.0f}")

# Aperiodic components
ax.loglog(*plot_noise1, lw=lw_noise, ls=":", zorder=1, label="1/f + noise")
ax.loglog(*plot_noise2, zorder=2, lw=lw_noise, ls=":", label="1/f + noise")

# Plot ground truth sim2
# ax.loglog(*plot_ground2, **line_ground, zorder=3, label=f"a={slope2:.0f}")

# Set axes
ax.set(**axes_b)
ax.legend()
# =============================================================================

# c)  =========================================================================
ax = axes[1, 0]

# Plot Sub 9 and 12
ax.loglog(*plot_sub9_signal, label="LFP Sub. 9")
ax.loglog(*plot_sub12_signal, label="LFP Sub. 12")

ax.loglog(*plot_sub9_plateau, label="Plateau")
ax.loglog(*plot_sub12_plateau)

# Plot Peak lines
ax.vlines(**line_osc1)
ax.vlines(**line_osc2)

# Plot Arrow left and right
ax.annotate(**arrow1)
ax.annotate(**arrow2)

# Annotate noise floor start as line
ax.annotate(**plateau_line9)
ax.annotate(**plateau_line12)

# Annotate noise floor start as text
ax.annotate(**plateau_txt9, fontsize=annotation_fontsize)
ax.annotate(**plateau_txt12, fontsize=annotation_fontsize)

# Set axes
ax.set(**axes_c)
ax.legend(loc=0)
# =============================================================================


# d)  =========================================================================
ax = axes[1, 1]

# Plot low noise subject
ax.loglog(*plot_sub_lowN_low, label="LFP low-noise")
ax.loglog(*plot_lowN_fit_sub, **line_fit,
          label=r"$a_{fit}$="f"{exp_lowN:.2f}")
ax.loglog(*plot_sub_lowN_high)
ax.loglog(*plot_sub_lowN_plateau)

# Plot low noise dummy
ax.loglog(*dummy_lowN, label="Dummy low-noise")
ax.loglog(*plot_lowN_fit_dummy, **line_fit,
          label=r"$a_{fit}$="f"{exp_lowN_d:.2f}")
ax.loglog(*plot_sub9, alpha=0.2, label="LFP Sub. 9")

# Plot Plateau lines
ax.vlines(**plateu_line_start)
ax.vlines(**plateu_line_end)

# Set axes
ax.set(**axes_d)
ax.legend()
# =============================================================================

# panel labels
for s, ax in zip("abcd", axes.flat):
    ax.text(s=s, **panel_labels, transform=ax.transAxes)

plt.tight_layout()
plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()
