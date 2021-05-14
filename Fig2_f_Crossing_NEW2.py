"""Figure 2 with updated osc function."""
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
from mne.time_frequency import psd_welch
import matplotlib.gridspec as gridspec


def osc_signals(slope=1, periodic_params=None, nlv=None, highpass=True,
                srate=2400, duration=180, seed=1):
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

    # Add white noise
    if nlv:
        w_noise = np.random.normal(scale=nlv, size=n_samples-2)
        noise += w_noise
        noise_osc += w_noise

    # Highpass filter
    if highpass:
        sos = sig.butter(4, 1, btype="hp", fs=srate, output='sos')
        noise = sig.sosfilt(sos, noise)
        noise_osc = sig.sosfilt(sos, noise_osc)

    return noise, noise_osc


# %% PARAMETERS

# Signal params
srate = 2400
nperseg = srate  # 4*srate too high resolution for fooof
welch_params = dict(fs=srate, nperseg=nperseg)

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig2_f_crossing.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# Colors

# a)
c_sim = "k"
c_error = "r"

c_range1 = "b"
c_range2 = "g"
c_range3 = "y"

# b)
c_real = "purple"

c_fit1 = c_real
c_fit2 = "c"
c_fit3 = "lime"
c_fit4 = "orange"

# c)
c_low = "deepskyblue"
c_med = "limegreen"
c_high = "orangered"

c_ground = "grey"


# %% a) Sim Toy Signal with Three Oscillations and Fit

# fit in all frequency ranges from 1 to 80...
lower_fitting_border = range(1, 80)
# ... to 100 Hz
upper_fitting_border = 100

# Oscillations parameters:
freq1, freq2, freq3 = 5, 15, 25  # Hz
amp1, amp2, amp3 = 5, 4, 1
width1, width2, width3 = 1, .1, 2
toy_slope = 1

periodic_params = [(freq1, amp1, width1),
                   (freq2, amp2, width2),
                   (freq3, amp3, width3)]

# Sim Toy Signal
_, toy_signal = osc_signals(slope=toy_slope, periodic_params=periodic_params)
freq, toy_psd = sig.welch(toy_signal, **welch_params)

# Filter 1-600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
toy_psd = toy_psd[filt]

# Fit fooof and subtract ground truth to obtain fitting error
fit_errors = []
fm = FOOOF(verbose=None)
for low in lower_fitting_border:
    freq_range = (low, upper_fitting_border)
    fm.fit(freq, toy_psd, freq_range)
    exp = fm.get_params("aperiodic", "exponent")
    error = np.abs(toy_slope - exp)
    fit_errors.append(error)


# %% B: Load and Fit

# Load data
data_path = "../data/Fig2/"
fname10 = "subj10_on_R8_raw.fif"

sub10 = mne.io.read_raw_fif(data_path + fname10, preload=True)

sub10.pick_channels(["STN_L23"])

# Notch Filter
filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}
sub10.notch_filter(**filter_params)

# Convert to numpy and calc PSD
start = int(0.5*srate)  # artefacts in beginning and end
stop = int(185*srate)
sub10 = sub10.get_data(start=start, stop=stop)[0]
freq, spec10 = sig.welch(sub10, **welch_params)

# Filter above highpass and below lowpass
freq = freq[filt]
spec10 = spec10[filt]

# Set common 1/f fitting ranges
frange1 = (1, 95)
frange2 = (30, 45)
frange3 = (40, 60)
frange4 = (1, 45)

# Set corresponding fooof fitting parameters
peak_width_limits = (1, 100)  # huge beta peak spans from 10 to almost 100 Hz
max_n_peaks = 0  # some fitting ranges try to avoid oscillations peaks
fooof_params1 = dict(peak_width_limits=peak_width_limits, verbose=False)
fooof_params2 = dict(max_n_peaks=max_n_peaks, verbose=False)
fooof_params3 = dict(max_n_peaks=max_n_peaks, verbose=False)
fooof_params4 = dict(peak_width_limits=peak_width_limits, verbose=False)

# Combine
fit_params = [(frange1, fooof_params1, c_fit1),
              (frange2, fooof_params2, c_fit2),
              (frange3, fooof_params3, c_fit3),
              (frange4, fooof_params4, c_fit4)]

# Fit for diferent ranges
fit_ranges = []
for frange, fooof_params, plot_color in fit_params:
    # fit
    fm = FOOOF(**fooof_params)
    fm.fit(freq, spec10, frange)
    ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
    plot_args = (fm.freqs, 10**ap_fit, plot_color)

    # set plot labels
    freq_low, freq_up = frange
    freq_str = f"{freq_low}-{freq_up}Hz"
    if freq_low == 1:  # add extra spaces if freq_low=1 for aligned legend
        freq_str = "  " + freq_str
    exp = fm.get_params("aperiodic", "exponent")
    plot_label = freq_str + f" a={exp:.2f}"
    plot_kwargs = dict(lw=3, ls="--", label=plot_label)

    # append plot argument (tuple) and plot_kwargs (dict) as tuple
    fit_ranges.append((plot_args, plot_kwargs))


# %% C: Reproduce PSD

nlv = 0.0003  # white noise level
slope = 1.5  # 1/f slope

# Oscillations as (frequency, amplitude, width)
alpha = (12, 1.7, 3)
low_beta = (18, 2, 2)
high_beta = (27, 20, 6)
gamma = (50, 6, 15)
HFO = (360, 20, 60)

oscillations = (alpha, low_beta, high_beta, gamma, HFO)

# Delta Oscillations
delta_freq = 2
delta_width = 6
low_delta = (delta_freq, 0, delta_width)
med_delta = (delta_freq, 1.9, delta_width)
high_delta = (delta_freq, 4.2, delta_width)

osc_params_low = [low_delta, *oscillations]
osc_params_med = [med_delta, *oscillations]
osc_params_high = [high_delta, *oscillations]

# Make signals
aperiodic, osc_low = osc_signals(slope=slope,
                                 periodic_params=osc_params_low,
                                 nlv=nlv)
aperiodic, osc_med = osc_signals(slope=slope,
                                 periodic_params=osc_params_med,
                                 nlv=nlv)
aperiodic, osc_high = osc_signals(slope=slope,
                                  periodic_params=osc_params_high,
                                  nlv=nlv)

# Calc PSD
freq, psd_aperiodic = sig.welch(aperiodic, **welch_params)
freq, psd_low = sig.welch(osc_low, **welch_params)
freq, psd_med = sig.welch(osc_med, **welch_params)
freq, psd_high = sig.welch(osc_high, **welch_params)

# Bandpass filter between 1Hz and 600Hz
freq = freq[filt]
psd_aperiodic = psd_aperiodic[filt]
psd_low = psd_low[filt]
psd_med = psd_med[filt]
psd_high = psd_high[filt]

# Normalize spectra to bring together real and simulated PSD.
# We cannot use the 1Hz offset because we want to show the impact of Delta
# oscillations -> so we normalize at the plateau by taking the median
# (to avoid notch filter outliers) and divide
plateau = (freq > 105) & (freq < 195)

spec10_adj = spec10 / np.median(spec10[plateau])
psd_aperiodic /= np.median(psd_aperiodic[plateau])
psd_low /= np.median(psd_low[plateau])
psd_med /= np.median(psd_med[plateau])
psd_high /= np.median(psd_high[plateau])

# Fit real and simulated spectra
fm_LFP = FOOOF(**fooof_params1)
fm_low = FOOOF(**fooof_params1)
fm_med = FOOOF(**fooof_params1)
fm_high = FOOOF(**fooof_params1)

fm_LFP.fit(freq, spec10_adj, frange1)
fm_low.fit(freq, psd_low, frange1)
fm_med.fit(freq, psd_med, frange1)
fm_high.fit(freq, psd_high, frange1)

exp_LFP = fm_LFP.get_params('aperiodic_params', 'exponent')
exp_low = fm_low.get_params('aperiodic_params', 'exponent')
exp_med = fm_med.get_params('aperiodic_params', 'exponent')
exp_high = fm_high.get_params('aperiodic_params', 'exponent')

ap_fit_LFP = gen_aperiodic(fm_LFP.freqs, fm_LFP.aperiodic_params_)
ap_fit_low = gen_aperiodic(fm_low.freqs, fm_low.aperiodic_params_)
ap_fit_med = gen_aperiodic(fm_med.freqs, fm_med.aperiodic_params_)
ap_fit_high = gen_aperiodic(fm_high.freqs, fm_high.aperiodic_params_)


# %% Plot params
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.size"] = 14

abc = dict(x=0, y=1.04, fontsize=20, fontdict=dict(fontweight="bold"))


# % Plot
fig = plt.figure(figsize=[9, 7.5], constrained_layout=True)

gs0 = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[10, 1.5, 10])

gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0])

ax1 = fig.add_subplot(gs00[0, 0])
ax2 = fig.add_subplot(gs00[1, 0])
ax3 = fig.add_subplot(gs00[:, 1])

gs01 = gs0[1]
ax_leg = fig.add_subplot(gs01)
ax_leg.axis("off")

gs02 = gs0[2].subgridspec(1, 3)

ax4 = fig.add_subplot(gs02[0])
ax5 = fig.add_subplot(gs02[1])
ax6 = fig.add_subplot(gs02[2])

# a)

ax = ax1
mask = (freq <= 100)
xticks_a = []
tick_dic = dict(xticks=xticks_a, xticklabels=xticks_a)
anno_dic = dict(x=100, ha="right", fontsize=9)
ax.loglog(freq[mask], toy_psd[mask], c_sim)
hline_height_log = (8e-8, 5.58e-9, 3.9e-10)
ax.hlines(hline_height_log[0], freq1, upper_fitting_border, color=c_range1, ls="--", label="Fit range 1")
ax.hlines(hline_height_log[1], freq2, upper_fitting_border, color=c_range2, ls="--", label="Fit range 2")
ax.hlines(hline_height_log[2], freq3, upper_fitting_border, color=c_range3, ls="--",  label="Fit range 3")
ax.text(s="Fitting range: 4-100Hz", y=1.1e-7, **anno_dic)
ax.text(s="11-100Hz", y=7.1e-9, **anno_dic)
ax.text(s="23-100Hz", y=5.5e-10, **anno_dic)
ax.text(s="a", **abc, transform=ax.transAxes)
xlim_a = (1, 126)
ax.set(**tick_dic, yticklabels=[], xlim=xlim_a,
       # xlabel="Frequency [Hz]",
       ylabel="PSD [a.u.]")

ax = ax2
ax.semilogx(lower_fitting_border, fit_errors, c_error)
ax.set(xlabel="Lower fitting range border [Hz]", ylabel="Fitting error")
hline_height = (1, .7, .4)
ax.hlines(hline_height[0], freq1, upper_fitting_border, color=c_range1, ls="--")
ax.hlines(hline_height[1], freq2, upper_fitting_border, color=c_range2, ls="--")
ax.hlines(hline_height[2], freq3, upper_fitting_border, color=c_range3, ls="--")
# xticks = [1, 4, 11, 23, 100]
xticks = [1, 10, 100]
yticks = [0, 1]
tick_dic = dict(xticks=xticks, xticklabels=xticks, yticks=yticks, xlim=xlim_a)
ax.set(**tick_dic)


# b)

ax = ax3
ax.loglog(freq, spec10, c=c_real)
for fit_range in fit_ranges:

    ax.loglog(*fit_range[0], **fit_range[1])
xticks = [1, 10, 100, 600]
yticks = [5e-3, 5e-2, 5e-1]
yticklabels = [5e-3, None, .5]
xlim_b = (1, 826)
ax.set(xlabel="Frequency [Hz]", xticks=xticks,
       xticklabels=xticks, yticks=yticks, yticklabels=yticklabels, xlim=xlim_b)
ax.set_ylabel(r"PSD [$\mu$$V^2$/Hz]", labelpad=-30)
ax.tick_params(axis="y", length=5, width=1.5)
leg = ax.legend(frameon=True, fontsize=10, bbox_to_anchor=(.55, 1.03))
leg.set_in_layout(False)
ax.text(s="b", **abc, transform=ax.transAxes)


# c)

ax = ax4

real = freq, spec10_adj
real_fit = fm_LFP.freqs, 10**ap_fit_LFP, "--"
real_kwargs = dict(c=c_real, alpha=.3, lw=2)

low = freq, psd_low, c_low
low_fit = fm_low.freqs, 10**ap_fit_low, "--"
low_kwargs = dict(c=c_low, lw=2)
x_arrow = 0.9
# x_arrow = 1
arr_pos_low = ("",
               (x_arrow, 10**ap_fit_low[0] * 0.95),
               (x_arrow, 10**ap_fit_LFP[0] * 1.1))
# =============================================================================
# arr_pos_low = ("",
#                (x_arrow, 10**ap_fit_low[0] * 1.),
#                (x_arrow, 10**ap_fit_LFP[0] * 1.))
# =============================================================================

med = freq, psd_med, c_med
med_fit = fm_med.freqs, 10**ap_fit_med, "--"
med_kwargs = dict(c=c_med, lw=2)
# arr_pos_med = "", (x_arrow, 10**ap_fit_med[0]), (x_arrow, 10**ap_fit_LFP[0])

high = freq, psd_high, c_high
high_fit = fm_high.freqs, 10**ap_fit_high, "--"
high_kwargs = dict(c=c_high, lw=2)
arr_pos_high = ("",
                (x_arrow, 10**ap_fit_high[0] * 1.1),
                (x_arrow, 10**ap_fit_LFP[0] * 1))
# =============================================================================
# arr_pos_high = ("",
#                 (x_arrow, 10**ap_fit_high[0] * 1.),
#                 (x_arrow, 10**ap_fit_LFP[0] * 1))
# =============================================================================

ground = freq, psd_aperiodic, c_ground
ground_kwargs = dict(lw=.5)

fill_mask = freq <= 4
fill_dic = dict(alpha=0.5)
tick_dic = dict(xticks=xticks, xticklabels=xticks, yticks=[])
arrow_dic = dict(arrowprops=dict(arrowstyle="->, "
                                 "head_length=0.2,head_width=0.2", lw=2))

ax.loglog(*real, **real_kwargs, label="STN-LFP")
ax.loglog(*real_fit, **real_kwargs, label=f"fooof LFP a={exp_LFP:.2f}")
ax.loglog(*ground, **ground_kwargs)
ax.loglog(*low)
ax.loglog(*low_fit, **low_kwargs, label=f"fooof sim1 a={exp_low:.2f}")
ax.fill_between(freq[fill_mask],
                psd_low[fill_mask], psd_aperiodic[fill_mask],
                color=c_low, **fill_dic)
ax.set_ylabel("PSD [a.u.]")
ax.set(**tick_dic)
ax.annotate(*arr_pos_low, **arrow_dic)
handles, labels = ax.get_legend_handles_labels()
ax.text(s="c", **abc, transform=ax.transAxes)


ax = ax5
ax.loglog(*real, **real_kwargs)
ax.loglog(*med)
ax.loglog(*ground, **ground_kwargs)
real_line, = ax.loglog(*real_fit, **real_kwargs)
ax.loglog(*med_fit, **med_kwargs, label=f"fooof sim2 a={exp_med:.2f}")
ax.fill_between(freq[fill_mask],
                psd_med[fill_mask], psd_aperiodic[fill_mask], color=c_med, **fill_dic)
ax.set_yticks([], minor=True)
ax.set_xlabel("Fitting range: 1-95 Hz")
ax.spines["left"].set_visible(False)
ax.set(**tick_dic)
# ax.annotate(*arr_pos_med, **arrow_dic)
hands, labs = ax.get_legend_handles_labels()
handles.extend(hands)
labels.extend(labs)


ax = ax6
ax.loglog(*real, **real_kwargs)
# ax.loglog(*high, **high_kwargs_k)
ax.loglog(*high)
real_line, = ax.loglog(*real_fit, **real_kwargs)
ax.loglog(*high_fit, **high_kwargs, label=f"fooof sim3 a={exp_high:.2f}")
ax.fill_between(freq[fill_mask],
                psd_high[fill_mask], psd_aperiodic[fill_mask],
                color=c_high, **fill_dic)
ax.loglog(*ground, **ground_kwargs, label="1/f + noise")
ax.set_yticks([], minor=True)
ax.spines["left"].set_visible(False)
ax.set(**tick_dic)
ax.annotate(*arr_pos_high, **arrow_dic)
hands, labs = ax.get_legend_handles_labels()
handles.extend(hands)
labels.extend(labs)
leg = ax_leg.legend(handles, labels, ncol=3, frameon=True, fontsize=12,
                    labelspacing=0.1, bbox_to_anchor=(.9, .7))
leg.set_in_layout(False)
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.show()


# =============================================================================
# NOT AS GOOD_ too much noise and change of alpha power necesssary
# # %% C: Reproduce PSD
# 
# nlv = 0.0003
# slope = 1.5
# # No oscillations
# periodic_params = [(2, .5, 1), (12, 7, 10), (18, 2, 5),
#                    (27, 30, 7), (55, 9, 15), (360, 30, 60)]
# 
# 
# # Make noise
# 
# aperiodic, pink1 = osc_signals(slope=slope, periodic_params=periodic_params,
#                            nlv=nlv)
# 
# freq, sim1 = sig.welch(pink1, fs=srate, nperseg=nperseg)
# freq, aperiodic = sig.welch(aperiodic, fs=srate, nperseg=nperseg)
# 
# # Bandpass filter between 1Hz and 600Hz
# filt = (freq > 0) & (freq <= 600)
# freq = freq[filt]
# sim1 = sim1[filt]
# aperiodic = aperiodic[filt]
# 
# # Adjust offset for real spectrum
# # cannot be normalized at 1Hz due to Delta Offset
# spec10 = spec10 / spec10[-1]
# sim1 /= sim1[-1]
# aperiodic /= aperiodic[-1]
# 
# # oscillations
# periodic_params = [(2, 2.5, 1), (12, 7, 10), (18, 2, 5),
#                    (27, 30, 7), (55, 9, 15), (360, 30, 60)]
# 
# 
# _, osc_high = osc_signals(slope=slope, periodic_params=periodic_params,
#                                  nlv=nlv)
# 
# freq, psd_high = sig.welch(osc_high, fs=srate, nperseg=nperseg)
# # Bandpass filter between 1Hz and 600Hz
# filt = (freq > 0) & (freq <= 600)
# freq = freq[filt]
# psd_high = psd_high[filt]
# 
# # Adjust offset for real spectrum
# psd_high /= psd_high[-1]
# 
# 
# # oscillations
# 
# periodic_params = [(2, 0, 1), (12, 3, 10), (18, 2, 5),
#                    (27, 30, 7), (55, 9, 15), (360, 30, 60)]
# 
# _, osc_low = osc_signals(slope=slope, periodic_params=periodic_params,
#                                 nlv=nlv)
# osc_low = osc_low
# 
# 
# 
# freq, sim1_deltaLow = sig.welch(osc_low, fs=srate, nperseg=nperseg)
# # Bandpass filter between 1Hz and 600Hz
# filt = (freq > 0) & (freq <= 600)
# freq = freq[filt]
# sim1_deltaLow = sim1_deltaLow[filt]
# 
# # Adjust offset for real spectrum
# psd_low = sim1_deltaLow / sim1_deltaLow[-1]
# 
# # Fit
# fm = FOOOF(**fit_params[0][1])
# fm.fit(freq, spec10, [1, 95])
# exp_LFP = fm.get_params('aperiodic_params', 'exponent')
# ap_fit_LFP = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
# 
# fm.fit(freq, psd_low, [1, 95])
# exp_low = fm.get_params('aperiodic_params', 'exponent')
# ap_fit_low = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
# 
# fm.fit(freq, sim1, [1, 95])
# exp_med = fm.get_params('aperiodic_params', 'exponent')
# ap_fit_med = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
# 
# fm.fit(freq, psd_high, [1, 95])
# exp_high = fm.get_params('aperiodic_params', 'exponent')
# ap_fit_high = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
# 
# 
# # % Plot
# mpl.rcParams["axes.spines.right"] = False
# mpl.rcParams["axes.spines.top"] = False
# mpl.rcParams["font.size"] = 14
# 
# abc = dict(x=0, y=1.04, fontsize=20, fontdict=dict(fontweight="bold"))
# 
# fig = plt.figure(figsize=[9, 7.5], constrained_layout=True)
# 
# gs0 = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[10, 1.5, 10])
# 
# gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0])
# 
# ax1 = fig.add_subplot(gs00[0, 0])
# ax2 = fig.add_subplot(gs00[1, 0])
# ax3 = fig.add_subplot(gs00[:, 1])
# 
# gs01 = gs0[1]
# ax_leg = fig.add_subplot(gs01)
# ax_leg.axis("off")
# 
# gs02 = gs0[2].subgridspec(1, 3)
# 
# ax4 = fig.add_subplot(gs02[0])
# ax5 = fig.add_subplot(gs02[1])
# ax6 = fig.add_subplot(gs02[2])
# 
# # a)
# 
# ax = ax1
# mask = (freq <= 100)
# xticks = []
# tick_dic = dict(xticks=xticks, xticklabels=xticks)
# anno_dic = dict(x=100, ha="right", fontsize=9)
# ax.loglog(freq[mask], noise_psd[mask], c_sim)
# hlines_y = [8e-8, 5.58e-9, 3.9e-10]
# ax.hlines(hlines_y[0], 4, 100, color=c_range1, ls="--", label="Fit range 1")
# ax.hlines(hlines_y[1], 11, 100, color=c_range2, ls="--", label="Fit range 2")
# ax.hlines(hlines_y[2], 23, 100, color=c_range3, ls="--",  label="Fit range 3")
# ax.text(s="Fitting range: 4-100Hz", y=1.1e-7, **anno_dic)
# ax.text(s="11-100Hz", y=7.1e-9, **anno_dic)
# ax.text(s="23-100Hz", y=5.5e-10, **anno_dic)
# ax.text(s="a", **abc, transform=ax.transAxes)
# ax.set(**tick_dic, yticklabels=[],
#        # xlabel="Frequency [Hz]",
#        ylabel="PSD [a.u.]")
# 
# ax = ax2
# ax.semilogx(lower_fitting_border, errors1, c_error)
# ax.set(xlabel="Lower fitting range border [Hz]", ylabel="Fitting error")
# ax.hlines(1, 4, 100, color=c_range1, ls="--")
# ax.hlines(.7, 11, 100, color=c_range2, ls="--")
# ax.hlines(.4, 23, 100, color=c_range3, ls="--")
# # xticks = [1, 4, 11, 23, 100]
# xticks = [1, 10, 100]
# yticks = [0, 1]
# tick_dic = dict(xticks=xticks, xticklabels=xticks,  yticks=yticks)
# ax.set(**tick_dic)
# 
# 
# # b)
# 
# ax = ax3
# ax.loglog(freq, spec10, c=c_real)
# for i in range(4):
#     fit = fits[i][0].freqs, 10**fits[i][2], fit_params[i][2]
#     freq1 = fit_params[i][0][0]
#     freq2 = fit_params[i][0][1]
#     if freq1 == 1:
#         freq_str = f"  {freq1}-{freq2}Hz"
#     else:
#         freq_str = f"{freq1}-{freq2}Hz"
#     kwargs = dict(lw=3, ls="--", label=freq_str + f" a={fits[i][1]:.2f}")
#     ax.loglog(*fit, **kwargs)
# xticks = [1, 10, 100, 600]
# yticks = [5e-3, 5e-2, 5e-1]
# yticklabels = [5e-3, None, .5]
# ax.set(xlabel="Frequency [Hz]", xticks=xticks,
#        xticklabels=xticks, yticks=yticks, yticklabels=yticklabels)
# ax.set_ylabel(r"PSD [$\mu$$V^2$/Hz]", labelpad=-30)
# ax.tick_params(axis="y", length=5, width=1.5)
# leg = ax.legend(frameon=True, fontsize=10, bbox_to_anchor=(.55, 1.03))
# leg.set_in_layout(False)
# ax.text(s="b", **abc, transform=ax.transAxes)
# 
# 
# # c)
# 
# ax = ax4
# 
# real = freq, spec10
# real_fit = fm.freqs, 10**ap_fit_LFP, "--"
# real_kwargs = dict(c=c_real, alpha=.3, lw=2)
# 
# low = freq, psd_low, c_low
# low_fit = fm.freqs, 10**ap_fit_low, "--"
# low_kwargs = dict(c=c_low, lw=2)
# x_arrow = 0.9
# # x_arrow = 1
# arr_pos_low = ("",
#                (x_arrow, 10**ap_fit_low[0] * 0.95),
#                (x_arrow, 10**ap_fit_LFP[0] * 1.1))
# # =============================================================================
# # arr_pos_low = ("",
# #                (x_arrow, 10**ap_fit_low[0] * 1.),
# #                (x_arrow, 10**ap_fit_LFP[0] * 1.))
# # =============================================================================
# 
# med = freq, sim1, c_med
# med_fit = fm.freqs, 10**ap_fit_med, "--"
# med_kwargs = dict(c=c_med, lw=2)
# # arr_pos_med = "", (x_arrow, 10**ap_fit_med[0]), (x_arrow, 10**ap_fit_LFP[0])
# 
# high = freq, psd_high, c_high
# high_fit = fm.freqs, 10**ap_fit_high, "--"
# high_kwargs = dict(c=c_high, lw=2)
# arr_pos_high = ("",
#                 (x_arrow, 10**ap_fit_high[0] * 1.1),
#                 (x_arrow, 10**ap_fit_LFP[0] * 1))
# # =============================================================================
# # arr_pos_high = ("",
# #                 (x_arrow, 10**ap_fit_high[0] * 1.),
# #                 (x_arrow, 10**ap_fit_LFP[0] * 1))
# # =============================================================================
# 
# ground = freq, aperiodic, c_ground
# ground_kwargs = dict(lw=.5)
# 
# fill_mask = freq <= 4
# fill_dic = dict(alpha=0.5)
# tick_dic = dict(xticks=xticks, xticklabels=xticks, yticks=[])
# arrow_dic = dict(arrowprops=dict(arrowstyle="->, "
#                                  "head_length=0.2,head_width=0.2", lw=2))
# 
# ax.loglog(*real, **real_kwargs, label="STN-LFP")
# ax.loglog(*real_fit, **real_kwargs, label=f"fooof LFP a={exp_LFP:.2f}")
# ax.loglog(*ground, **ground_kwargs)
# ax.loglog(*low)
# ax.loglog(*low_fit, **low_kwargs, label=f"fooof sim1 a={exp_low:.2f}")
# ax.fill_between(freq[fill_mask],
#                 psd_low[fill_mask], aperiodic[fill_mask],
#                 color=c_low, **fill_dic)
# ax.set_ylabel("PSD [a.u.]")
# ax.set(**tick_dic)
# ax.annotate(*arr_pos_low, **arrow_dic)
# handles, labels = ax.get_legend_handles_labels()
# ax.text(s="c", **abc, transform=ax.transAxes)
# 
# 
# ax = ax5
# ax.loglog(*real, **real_kwargs)
# ax.loglog(*med)
# ax.loglog(*ground, **ground_kwargs)
# real_line, = ax.loglog(*real_fit, **real_kwargs)
# ax.loglog(*med_fit, **med_kwargs, label=f"fooof sim2 a={exp_med:.2f}")
# ax.fill_between(freq[fill_mask],
#                 sim1[fill_mask], aperiodic[fill_mask], color=c_med, **fill_dic)
# ax.set_yticks([], minor=True)
# ax.set_xlabel("Fitting range: 1-95 Hz")
# ax.spines["left"].set_visible(False)
# ax.set(**tick_dic)
# # ax.annotate(*arr_pos_med, **arrow_dic)
# hands, labs = ax.get_legend_handles_labels()
# handles.extend(hands)
# labels.extend(labs)
# 
# 
# ax = ax6
# ax.loglog(*real, **real_kwargs)
# # ax.loglog(*high, **high_kwargs_k)
# ax.loglog(*high)
# real_line, = ax.loglog(*real_fit, **real_kwargs)
# ax.loglog(*high_fit, **high_kwargs, label=f"fooof sim3 a={exp_high:.2f}")
# ax.fill_between(freq[fill_mask],
#                 psd_high[fill_mask], aperiodic[fill_mask],
#                 color=c_high, **fill_dic)
# ax.loglog(*ground, **ground_kwargs, label="1/f + noise")
# ax.set_yticks([], minor=True)
# ax.spines["left"].set_visible(False)
# ax.set(**tick_dic)
# ax.annotate(*arr_pos_high, **arrow_dic)
# hands, labs = ax.get_legend_handles_labels()
# handles.extend(hands)
# labels.extend(labs)
# leg = ax_leg.legend(handles, labels, ncol=3, frameon=True, fontsize=12,
#                     labelspacing=0.1, bbox_to_anchor=(.9, .7))
# leg.set_in_layout(False)
# for lh in leg.legendHandles:
#     lh.set_alpha(1)
# plt.show()
# 
# 
# =============================================================================
