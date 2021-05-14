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

# Signal
srate = 2400
nperseg = srate  # 4*srate too high resolution for fooof

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


# %% a) Sim Signal and fit

# fit in all frequency ranges from 1 to 80...
lower_fitting_border = np.arange(0, 80, 1)
# ... to 100 Hz
upper_fitting_border = 100

# Oscillations parameters:
osc_freq1, osc_freq2, osc_freq3 = 5, 15, 25
amp1, amp2, amp3 = 5, 4, 1
width1, width2, width3 = 1, .1, 2

periodic_params = [(osc_freq1, amp1, width1),
                   (osc_freq2, amp2, width2),
                   (osc_freq3, amp3, width3)]

signals = osc_signals(periodic_params=periodic_params)
freq, noise_psd = sig.welch(signals[1], fs=srate, nperseg=nperseg)

# Filter 1-600Hz
freq = freq[1:601]
noise_psd = noise_psd[1:601]

# Calc fooof vary freq ranges
errors1 = []
for low in lower_fitting_border:
    fm = FOOOF(verbose=None)
    fm.fit(freq, noise_psd, [low, upper_fitting_border])
    exp = fm.get_params("aperiodic", "exponent")
    error = 1 - exp
    error = np.abs(error)
    errors1.append(error)


# %% B: Load and fit
ch = 'STN_L23'

# Load data
path = "../data/Fig2/"
fname10_on = "subj10_on_R8_raw.fif"

sub10_on = mne.io.read_raw_fif(path + fname10_on, preload=True)

sub10_on.pick_channels([ch])

filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}

sub10_on.notch_filter(**filter_params)

welch_params = {"fmin": 1,
                "fmax": 600,
                "tmin": 0.5,
                "tmax": 185,
                "n_fft": srate,
                "n_overlap": srate // 2,
                "average": "mean"}

spec10_on, freq = psd_welch(sub10_on, **welch_params)
spec10_on = spec10_on[0]

frange1 = (1, 95)  # 96 -> a = 0.70
frange2 = (30, 45)
frange3 = (40, 60)
frange4 = (1, 45)

fit_params = [(frange1, dict(peak_width_limits=[1, 100], verbose=False), c_fit1),
              # (frange2, dict(peak_width_limits=[1, 100]), c_fit2),
              (frange2, dict(max_n_peaks=0, verbose=False), c_fit2),
              (frange3, dict(max_n_peaks=0, verbose=False), c_fit3),
              (frange4, dict(peak_width_limits=[1, 100], verbose=False), c_fit4)]

fits = []
for i in range(4):
    fm = FOOOF(**fit_params[i][1])
    fm.fit(freq, spec10_on, fit_params[i][0])
    exp = fm.aperiodic_params_[1]
    ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
    fits.append((fm, exp, ap_fit))


# %% C: Reproduce PSD

nlv = 0.00026
#nlv = 0.0003
slope = 1.5
# No oscillations
periodic_params = [(2, 3.1, 6), (12, 1, 2.5), (18, 1, 2),
                   (27, 26, 7), (55, 8, 20), (360, 25, 60)]


# Make noise

pure1, pink1 = osc_signals(slope=slope, periodic_params=periodic_params,
                           nlv=nlv)

freq, sim1 = sig.welch(pink1, fs=srate, nperseg=nperseg)
freq, pure1 = sig.welch(pure1, fs=srate, nperseg=nperseg)

# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
sim1 = sim1[filt]
pure1 = pure1[filt]

# Adjust offset for real spectrum
# cannot be normalized at 1Hz due to Delta Offset
spec10_on_adj = spec10_on / spec10_on[-1]
sim1 /= sim1[-1]
pure1 /= pure1[-1]

# oscillations
periodic_params = [(2, 5.3, 6), (12, 1, 2.5), (18, 1, 2),
                   (27, 26, 7), (55, 8, 20), (360, 25, 60)]


_, pink1_deltaHigh = osc_signals(slope=slope, periodic_params=periodic_params,
                                 nlv=nlv)

freq, sim1_deltaHigh = sig.welch(pink1_deltaHigh, fs=srate, nperseg=nperseg)
# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
sim1_deltaHigh = sim1_deltaHigh[filt]

# Adjust offset for real spectrum
sim1_deltaHigh /= sim1_deltaHigh[-1]


# oscillations

periodic_params = [(2, 1.5, 6), (12, 1, 2.5), (18, 1, 2),
                   (27, 26, 7), (55, 8, 20), (360, 25, 60)]

_, pink1_deltaLow = osc_signals(slope=slope, periodic_params=periodic_params,
                                nlv=nlv)
pink1_deltaLow = pink1_deltaLow



freq, sim1_deltaLow = sig.welch(pink1_deltaLow, fs=srate, nperseg=nperseg)
# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
sim1_deltaLow = sim1_deltaLow[filt]

# Adjust offset for real spectrum
sim1_deltaLow_adj = sim1_deltaLow / sim1_deltaLow[-1]

# Fit
fm = FOOOF(**fit_params[0][1])
fm.fit(freq, spec10_on_adj, [1, 95])
exp_LFP = fm.get_params('aperiodic_params', 'exponent')
ap_fit_LFP = gen_aperiodic(fm.freqs, fm.aperiodic_params_)

fm.fit(freq, sim1_deltaLow_adj, [1, 95])
exp_low = fm.get_params('aperiodic_params', 'exponent')
ap_fit_low = gen_aperiodic(fm.freqs, fm.aperiodic_params_)

fm.fit(freq, sim1, [1, 95])
exp_med = fm.get_params('aperiodic_params', 'exponent')
ap_fit_med = gen_aperiodic(fm.freqs, fm.aperiodic_params_)

fm.fit(freq, sim1_deltaHigh, [1, 95])
exp_high = fm.get_params('aperiodic_params', 'exponent')
ap_fit_high = gen_aperiodic(fm.freqs, fm.aperiodic_params_)


# %% Plot
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.size"] = 14

abc = dict(x=0, y=1.04, fontsize=20, fontdict=dict(fontweight="bold"))

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
xticks = []
tick_dic = dict(xticks=xticks, xticklabels=xticks)
anno_dic = dict(x=100, ha="right", fontsize=9)
ax.loglog(freq[mask], noise_psd[mask], c_sim)
hlines_y = [8e-8, 5.58e-9, 3.9e-10]
ax.hlines(hlines_y[0], 4, 100, color=c_range1, ls="--", label="Fit range 1")
ax.hlines(hlines_y[1], 11, 100, color=c_range2, ls="--", label="Fit range 2")
ax.hlines(hlines_y[2], 23, 100, color=c_range3, ls="--",  label="Fit range 3")
ax.text(s="Fitting range: 4-100Hz", y=1.1e-7, **anno_dic)
ax.text(s="11-100Hz", y=7.1e-9, **anno_dic)
ax.text(s="23-100Hz", y=5.5e-10, **anno_dic)
ax.text(s="a", **abc, transform=ax.transAxes)
ax.set(**tick_dic, yticklabels=[],
       # xlabel="Frequency [Hz]",
       ylabel="PSD [a.u.]")

ax = ax2
ax.semilogx(lower_fitting_border, errors1, c_error)
ax.set(xlabel="Lower fitting range border [Hz]", ylabel="Fitting error")
ax.hlines(1, 4, 100, color=c_range1, ls="--")
ax.hlines(.7, 11, 100, color=c_range2, ls="--")
ax.hlines(.4, 23, 100, color=c_range3, ls="--")
# xticks = [1, 4, 11, 23, 100]
xticks = [1, 10, 100]
yticks = [0, 1]
tick_dic = dict(xticks=xticks, xticklabels=xticks,  yticks=yticks)
ax.set(**tick_dic)


# b)

ax = ax3
ax.loglog(freq, spec10_on, c=c_real)
for i in range(4):
    fit = fits[i][0].freqs, 10**fits[i][2], fit_params[i][2]
    freq1 = fit_params[i][0][0]
    freq2 = fit_params[i][0][1]
    if freq1 == 1:
        freq_str = f"  {freq1}-{freq2}Hz"
    else:
        freq_str = f"{freq1}-{freq2}Hz"
    kwargs = dict(lw=3, ls="--", label=freq_str + f" a={fits[i][1]:.2f}")
    ax.loglog(*fit, **kwargs)
xticks = [1, 10, 100, 600]
yticks = [5e-3, 5e-2, 5e-1]
yticklabels = [5e-3, None, .5]
ax.set(xlabel="Frequency [Hz]", xticks=xticks,
       xticklabels=xticks, yticks=yticks, yticklabels=yticklabels)
ax.set_ylabel(r"PSD [$\mu$$V^2$/Hz]", labelpad=-30)
ax.tick_params(axis="y", length=5, width=1.5)
leg = ax.legend(frameon=True, fontsize=10, bbox_to_anchor=(.55, 1.03))
leg.set_in_layout(False)
ax.text(s="b", **abc, transform=ax.transAxes)


# c)

ax = ax4

real = freq, spec10_on_adj
real_fit = fm.freqs, 10**ap_fit_LFP, "--"
real_kwargs = dict(c=c_real, alpha=.3, lw=2)

low = freq, sim1_deltaLow_adj, c_low
low_fit = fm.freqs, 10**ap_fit_low, "--"
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

med = freq, sim1, c_med
med_fit = fm.freqs, 10**ap_fit_med, "--"
med_kwargs = dict(c=c_med, lw=2)
# arr_pos_med = "", (x_arrow, 10**ap_fit_med[0]), (x_arrow, 10**ap_fit_LFP[0])

high = freq, sim1_deltaHigh, c_high
high_fit = fm.freqs, 10**ap_fit_high, "--"
high_kwargs = dict(c=c_high, lw=2)
arr_pos_high = ("",
                (x_arrow, 10**ap_fit_high[0] * 1.1),
                (x_arrow, 10**ap_fit_LFP[0] * 1))
# =============================================================================
# arr_pos_high = ("",
#                 (x_arrow, 10**ap_fit_high[0] * 1.),
#                 (x_arrow, 10**ap_fit_LFP[0] * 1))
# =============================================================================

ground = freq, pure1, c_ground
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
                sim1_deltaLow_adj[fill_mask], pure1[fill_mask],
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
                sim1[fill_mask], pure1[fill_mask], color=c_med, **fill_dic)
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
                sim1_deltaHigh[fill_mask], pure1[fill_mask],
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
# pure1, pink1 = osc_signals(slope=slope, periodic_params=periodic_params,
#                            nlv=nlv)
# 
# freq, sim1 = sig.welch(pink1, fs=srate, nperseg=nperseg)
# freq, pure1 = sig.welch(pure1, fs=srate, nperseg=nperseg)
# 
# # Bandpass filter between 1Hz and 600Hz
# filt = (freq > 0) & (freq <= 600)
# freq = freq[filt]
# sim1 = sim1[filt]
# pure1 = pure1[filt]
# 
# # Adjust offset for real spectrum
# # cannot be normalized at 1Hz due to Delta Offset
# spec10_on_adj = spec10_on / spec10_on[-1]
# sim1 /= sim1[-1]
# pure1 /= pure1[-1]
# 
# # oscillations
# periodic_params = [(2, 2.5, 1), (12, 7, 10), (18, 2, 5),
#                    (27, 30, 7), (55, 9, 15), (360, 30, 60)]
# 
# 
# _, pink1_deltaHigh = osc_signals(slope=slope, periodic_params=periodic_params,
#                                  nlv=nlv)
# 
# freq, sim1_deltaHigh = sig.welch(pink1_deltaHigh, fs=srate, nperseg=nperseg)
# # Bandpass filter between 1Hz and 600Hz
# filt = (freq > 0) & (freq <= 600)
# freq = freq[filt]
# sim1_deltaHigh = sim1_deltaHigh[filt]
# 
# # Adjust offset for real spectrum
# sim1_deltaHigh /= sim1_deltaHigh[-1]
# 
# 
# # oscillations
# 
# periodic_params = [(2, 0, 1), (12, 3, 10), (18, 2, 5),
#                    (27, 30, 7), (55, 9, 15), (360, 30, 60)]
# 
# _, pink1_deltaLow = osc_signals(slope=slope, periodic_params=periodic_params,
#                                 nlv=nlv)
# pink1_deltaLow = pink1_deltaLow
# 
# 
# 
# freq, sim1_deltaLow = sig.welch(pink1_deltaLow, fs=srate, nperseg=nperseg)
# # Bandpass filter between 1Hz and 600Hz
# filt = (freq > 0) & (freq <= 600)
# freq = freq[filt]
# sim1_deltaLow = sim1_deltaLow[filt]
# 
# # Adjust offset for real spectrum
# sim1_deltaLow_adj = sim1_deltaLow / sim1_deltaLow[-1]
# 
# # Fit
# fm = FOOOF(**fit_params[0][1])
# fm.fit(freq, spec10_on_adj, [1, 95])
# exp_LFP = fm.get_params('aperiodic_params', 'exponent')
# ap_fit_LFP = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
# 
# fm.fit(freq, sim1_deltaLow_adj, [1, 95])
# exp_low = fm.get_params('aperiodic_params', 'exponent')
# ap_fit_low = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
# 
# fm.fit(freq, sim1, [1, 95])
# exp_med = fm.get_params('aperiodic_params', 'exponent')
# ap_fit_med = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
# 
# fm.fit(freq, sim1_deltaHigh, [1, 95])
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
# ax.loglog(freq, spec10_on, c=c_real)
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
# real = freq, spec10_on_adj
# real_fit = fm.freqs, 10**ap_fit_LFP, "--"
# real_kwargs = dict(c=c_real, alpha=.3, lw=2)
# 
# low = freq, sim1_deltaLow_adj, c_low
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
# high = freq, sim1_deltaHigh, c_high
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
# ground = freq, pure1, c_ground
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
#                 sim1_deltaLow_adj[fill_mask], pure1[fill_mask],
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
#                 sim1[fill_mask], pure1[fill_mask], color=c_med, **fill_dic)
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
#                 sim1_deltaHigh[fill_mask], pure1[fill_mask],
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
