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
lower_fitting_borders = range(1, 80)
# ... to 100 Hz
upper_fitting_border = 100

# Oscillations parameters:
freq1, freq2, freq3 = 5, 15, 25  # Hz
amp1, amp2, amp3 = 5, 2.5, 1
width1, width2, width3 = 1, .1, 2
toy_slope = 1

periodic_params = [(freq1, amp1, width1),
                   (freq2, amp2, width2),
                   (freq3, amp3, width3)]

# Sim Toy Signal
_, toy_signal = osc_signals(slope=toy_slope, periodic_params=periodic_params)
freq, toy_psd = sig.welch(toy_signal, **welch_params)

# Filter 1-100Hz
filt = (freq > 0) & (freq <= 100)
freq = freq[filt]
toy_psd = toy_psd[filt]

toy_plot = (freq, toy_psd, c_sim)

# Fit fooof and subtract ground truth to obtain fitting error
fit_errors = []
fm = FOOOF(verbose=None)
for low in lower_fitting_borders:
    freq_range = (low, upper_fitting_border)
    fm.fit(freq, toy_psd, freq_range)
    exp = fm.get_params("aperiodic", "exponent")
    error = np.abs(toy_slope - exp)
    fit_errors.append(error)

error_plot = (lower_fitting_borders, fit_errors, c_error)

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
filt = (freq > 0) & (freq <= 600)

freq = freq[filt]
spec10 = spec10[filt]

plot_psd_spec10 = (freq, spec10, c_real)

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

# Create tuples for plotting
plot_psd_spec10_adj = (freq, spec10_adj)
plot_psd_low = (freq, psd_low, c_low)
plot_psd_med = (freq, psd_med, c_med)
plot_psd_high = (freq, psd_high, c_high)

# Summarize
psd_delta_vary = [plot_psd_low, plot_psd_med, plot_psd_high]

# Plot delta power
delta_mask = (freq <= 4)

freq_delta = freq[delta_mask]
psd_aperiodic_delta = psd_aperiodic[delta_mask]
psd_low_delta = psd_low[delta_mask]
psd_med_delta = psd_med[delta_mask]
psd_high_delta = psd_high[delta_mask]

plot_delta_low = (freq_delta, psd_low_delta, psd_aperiodic_delta)
plot_delta_med = (freq_delta, psd_med_delta, psd_aperiodic_delta)
plot_delta_high = (freq_delta, psd_high_delta, psd_aperiodic_delta)

# Summarize
delta_power = [plot_delta_low, plot_delta_med, plot_delta_high]


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

# Summarize
exponents = [exp_low, exp_med, exp_high]
delta_labels = [f"fooof sim{i} a={exp:.2f}" for i, exp in enumerate(exponents)]


ap_fit_LFP = gen_aperiodic(fm_LFP.freqs, fm_LFP.aperiodic_params_)
ap_fit_low = gen_aperiodic(fm_low.freqs, fm_low.aperiodic_params_)
ap_fit_med = gen_aperiodic(fm_med.freqs, fm_med.aperiodic_params_)
ap_fit_high = gen_aperiodic(fm_high.freqs, fm_high.aperiodic_params_)

# Create tuples for plotting
plot_fit_spec10 = (fm_LFP.freqs, 10**ap_fit_LFP, "--")
plot_aperiodic = (freq, psd_aperiodic, c_ground)
plot_fit_low = (fm_low.freqs, 10**ap_fit_low, "--")
plot_fit_med = (fm_med.freqs, 10**ap_fit_med, "--")
plot_fit_high = (fm_high.freqs, 10**ap_fit_high, "--")

# Summarize
psd_delta_fits = [plot_fit_low, plot_fit_med, plot_fit_high]


spec10_kwargs = dict(c=c_real, alpha=.3, lw=2)
aperiodic_kwargs = dict(lw=.5)
low_kwargs = dict(c=c_low, lw=2)
med_kwargs = dict(c=c_med, lw=2)
high_kwargs = dict(c=c_high, lw=2)

# Summarize
delta_kwargs = [low_kwargs, med_kwargs, high_kwargs]
colors_c = [c_low, c_med, c_high]


# %% Plot params

width = 7.25  # inches
panel_fontsize = 12
legend_fontsize1 = 9
legend_fontsize2 = 10
label_fontsize = 10
tick_fontsize = 9
annotation_fontsize = 9

mpl.rcParams['xtick.labelsize'] = tick_fontsize
mpl.rcParams['ytick.labelsize'] = tick_fontsize
mpl.rcParams['axes.labelsize'] = label_fontsize
mpl.rcParams['legend.fontsize'] = legend_fontsize1
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.size"] = 14


abc = dict(x=0, y=1.04, fontsize=panel_fontsize,
           fontdict=dict(fontweight="bold"))

# a)
# a1
xticklabels_a1 = []
yticks_a = [1e-10, 1e-7]
yticklabels_a1 = []
xlim_a = (1, 100)
ylabel_a1 = "PSD [a.u.]"
labelpad = 5
axes_a1 = dict(xticklabels=xticklabels_a1, yticks=yticks_a,
               yticklabels=yticklabels_a1, xlim=xlim_a)
freqs123 = [freq1, freq2, freq3]
colors123 = [c_range1, c_range2, c_range3]
hline_height_log = (8e-8, 5.58e-9, 3.9e-10)
text_height_log = (1.1e-7, 7.1e-9, 5.5e-10)
text_dic = dict(x=100, ha="right", fontsize=annotation_fontsize)

# a2
xticks_a2 = [1, 10, 100]
yticks_a2 = [0, 1]
xlabel_a2 = "Lower fitting range border [Hz]"
ylabel_a2 = "Fitting error"
axes_a2 = dict(xticks=xticks_a2, xticklabels=xticks_a2, yticks=yticks_a2,
               xlim=xlim_a, xlabel=xlabel_a2)
hline_height = (1, .7, .4)

# b)
xticks_b = [1, 10, 100, 600]
# yticks_b = [5e-3, 5e-2, 5e-1]
# yticklabels_b = [r"5$\cdot10^{-3}$", r"5$\cdot10^{-2}$", r"5$\cdot10^{-1}$"]
xlim_b = (1, 826)
xlabel_b = "Frequency [Hz]"
ylabel_b = r"PSD [$\mu$$V^2$/Hz]"
axes_b = dict(xlabel=xlabel_b, xticks=xticks_b, xticklabels=xticks_b,
              # yticks=yticks_b, yticklabels=yticklabels_b,
              xlim=xlim_b, ylabel=ylabel_b)

# c)
ylabel_c = "PSD [a.u.]"
axes_c = dict(xticks=xticks_b, xticklabels=xticks_b,
              yticks=[])
x_label_c2 = f"Fitting range: {frange1[0]}-{frange1[1]} Hz"
leg_c = dict(ncol=3, loc=10, fontsize=legend_fontsize2,
             bbox_to_anchor=(.5, -.7))
delta_fill_dic = dict(alpha=0.5)

# Annotate increased/decreased delta power with arrows
x_arrow = 0.9  # set arrow slightly below 1Hz
spec10_fit_offset = 10**ap_fit_LFP[0]  # LFP power at 1Hz
low_fit_offset = 10**ap_fit_low[0]  # sim low power at 1Hz
high_fit_offset = 10**ap_fit_high[0]  # sim high power at 1Hz

# Arrow coordinates. Up-scale to account for x_arrow < 1 Hz
arr_head_low = (x_arrow, low_fit_offset * 0.95)
arr_tail_low = (x_arrow, spec10_fit_offset * 1.1)

arr_head_high = (x_arrow, high_fit_offset * 1.1)
arr_tail_high = (x_arrow, spec10_fit_offset * 1)

# Arrow style
arrowstyle = "->, head_length=0.2,head_width=0.2"
arrow_dic = dict(arrowprops=dict(arrowstyle=arrowstyle, lw=2))

# Arrow dic
arr_pos_low = dict(text="", xy=arr_head_low, xytext=arr_tail_low,
                   **arrow_dic)
arr_pos_high = dict(text="", xy=arr_head_high, xytext=arr_tail_high,
                    **arrow_dic)


# %% Plot

# Prepare Gridspec
fig = plt.figure(figsize=[8, width], constrained_layout=True)

gs0 = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[10, 1, 10])

# a) and b)
gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0],
                                        width_ratios=[8, 10])
ax1 = fig.add_subplot(gs00[0, 0])
ax2 = fig.add_subplot(gs00[1, 0])
ax3 = fig.add_subplot(gs00[:, 1])

# Legend suplot
gs01 = gs0[1]
ax_leg = fig.add_subplot(gs01)
ax_leg.axis("off")

# c)
gs02 = gs0[2].subgridspec(1, 3)
ax4 = fig.add_subplot(gs02[0])
ax5 = fig.add_subplot(gs02[1])
ax6 = fig.add_subplot(gs02[2])

c_axes = [ax4, ax5, ax6]

# a)
# a1
ax = ax1

# Plot sim
ax.loglog(*toy_plot)

# Annotate fitting ranges
for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
    y = hline_height_log[i]
    xmin = freq_low
    xmax = upper_fitting_border
    coords = (y, xmin, xmax)
    hline_dic = dict(color=color, ls="--")
    ax.hlines(*coords, **hline_dic)
    s = f"{freq_low}-{xmax}Hz"
    if i == 0:
        s = "Fitting range: " + s
    y = text_height_log[i]
    ax.text(s=s, y=y, **text_dic)

# Set axes
ax.text(s="a", **abc, transform=ax.transAxes)
ax.set(**axes_a1)
ax.set_ylabel(ylabel_a1, labelpad=labelpad)

# a2
ax = ax2

# Plot error
ax.semilogx(*error_plot)

# Annotate fitting ranges
for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
    y = hline_height[i]
    xmin = freq_low
    xmax = upper_fitting_border
    coords = (y, xmin, xmax)
    hline_dic = dict(color=color, ls="--")
    ax.hlines(*coords, **hline_dic)

# Set axes
ax.set(**axes_a2)
ax.set_ylabel(ylabel_a2, labelpad=0)


# b)
ax = ax3

# Plot spectrum
ax.loglog(*plot_psd_spec10)

# Plot fooof fits
for fit_range in fit_ranges:
    ax.loglog(*fit_range[0], **fit_range[1])

# Set axes
ax.set(**axes_b)
ax.legend()
ax.text(s="b", **abc, transform=ax.transAxes)


# c)

# Make sure we have just one label for each repetitive plot
spec10_label = ["STN-LFP", None, None]
spec10_fit_label = [f"fooof LFP a={exp_LFP:.2f}", None, None]
aperiodic_label = [None, None, "1/f + noise"]

arrows = [arr_pos_low, None, arr_pos_high]

for i, ax in enumerate(c_axes):

    # Plot LFP and fooof fit
    ax.loglog(*plot_psd_spec10_adj, **spec10_kwargs, label=spec10_label[i])
    ax.loglog(*plot_fit_spec10, **spec10_kwargs, label=spec10_fit_label[i])

    # Plot aperiodic component of sim
    ax.loglog(*plot_aperiodic, **aperiodic_kwargs, label=aperiodic_label[i])

    # Plot sim low delta power and fooof fit
    ax.loglog(*psd_delta_vary[i])
    ax.loglog(*psd_delta_fits[i], **delta_kwargs[i], label=delta_labels[i])

    # Indicate delta power as fill between aperiodic component
    # and full spectrum
    ax.fill_between(*delta_power[i], color=colors_c[i], **delta_fill_dic)

    # Draw arrow
    if i != 1:
        ax.annotate(**arrows[i])
    else:
        ax.set_xlabel(x_label_c2)

    # Save legend handles labels and set axes
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.set_ylabel(ylabel_c, labelpad=labelpad)
        ax.text(s="c", **abc, transform=ax.transAxes)
    else:
        hands, labs = ax.get_legend_handles_labels()
        handles.extend(hands)
        labels.extend(labs)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([], minor=True)
    ax.set(**axes_c)

# Set legend between subplots
leg = ax_leg.legend(handles, labels, **leg_c)
leg.set_in_layout(False)
for lh in leg.legendHandles:
    lh.set_alpha(1)

plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()

# %% Plot Supp Mat b

"""
Make supp. fooof fits for b) and c) and maybe one example of a)
"""

# =============================================================================
# fig, axes = plt.subplots(2, 4, figsize=[16, 8])
# for i in range(4):
#     ax = axes[0, i]
#     kwargs = dict(add_legend=False,
#                   aperiodic_kwargs=dict(color=fit_params[i][2], alpha=1),
#                   data_kwargs=dict(color=c_real))
#     title = f"{fit_params[i][0][0]}-{fit_params[i][0][1]}Hz"
#     fits[i][0].plot(ax=ax, plt_log=True, **kwargs)
#     ax.set_title(title, fontsize=30)
#     ax.spines['right'].set_visible(True)
#     ax.spines['top'].set_visible(True)
#     if i > 0:
#         ax.set_ylabel("")
# 
#     ax = axes[1, i]
#     fits[i][0].plot(ax=ax, plt_log=False, **kwargs)
#     if i > 0:
#         ax.set_ylabel("")
#     ax.spines['right'].set_visible(True)
#     ax.spines['top'].set_visible(True)
# plt.tight_layout()
# plt.savefig(fig_path + fig_name[:-4] + "Supp.pdf", bbox_inches="tight")
# plt.show()
# =============================================================================
