"""
Very broad peak widths require very large resampling factors.

Panel a, b, c:
    - same as panel a Fig 4 for increasing peak widths!
    - include aperiodic components

d, e, f:
    - real data with small, medium, and large peak widths and different
    resampling factors
    - large peak: use LFP data of Esther
    - medium peak: use MEG source data of Esther
    - small peak:
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig
from pathlib import Path
# import mne
# from fooof import FOOOF
# from fooof.sim.gen import gen_aperiodic
import matplotlib.gridspec as gridspec
from noise_helper import irasa
try:
    from tqdm import trange
except ImportError:
    trange = range


def osc_signals(slope, periodic_params=None, nlv=None, highpass=True,
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


def calc_error(signal):
    """Fit IRASA and subtract ground truth to obtain fitting error."""
    fit_errors = []
    for i in trange(len(lower_fitting_borders)):
        freq_range = (lower_fitting_borders[i], upper_fitting_border)
        _, _, _, params = irasa(data=signal, band=freq_range, sf=srate)
        exp = -params["Slope"][0]
        error = np.abs(toy_slope - exp)
        fit_errors.append(error)
    return fit_errors


# %% PARAMETERS

# Signal params
srate = 2400
nperseg = srate  # 4*srate too high resolution for fooof
welch_params = dict(fs=srate, nperseg=nperseg)

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig5_IRASA_PeakWidth.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# Litvak file names
path = "../data/Fig5/"
# fname = "subj6_off_R1_raw.fif"

# Colors

# a)
c_sim = "k"
c_error = "r"
c_noise = "darkgray"

c_range1 = "b"
c_range2 = "g"
c_range3 = "y"

c_ap = "grey"

# b)
c_real = "purple"

# c_fit3 = "lime"

# c)
c_fooof = "deepskyblue"
c_IRASA = "orangered"
# c_med = "limegreen"

lw = 2

# %% a Sim Toy Signal with Three Oscillations and Fit

# fit in all frequency ranges from 1 to 80...
lower_fitting_borders = range(1, 80)
# ... to 100 Hz
upper_fitting_border = 100

# Oscillations parameters:
toy_slope = 2
freq1, freq2, freq3 = 5, 15, 35  # Hz
amp = 1
width = 1

periodic_params_a = [(freq1, amp*2, width*.2),
                     (freq2, amp*.4, width*.25),
                     (freq3, amp*.35, width*1.8)]

periodic_params_b = [(freq1, amp*2.8, width*.42),
                     (freq2, amp*1, width*.84),
                     (freq3, amp*.6, width*2)]

periodic_params_c = [(freq1, amp*3.5, width*.6),
                     (freq2, amp*1.7, width*1.2),
                     (freq3, amp*.7, width*2.5)]
# Sim Toy Signal
_, toy_signal_a = osc_signals(toy_slope, periodic_params=periodic_params_a,
                              highpass=False)
_, toy_signal_b = osc_signals(toy_slope, periodic_params=periodic_params_b,
                              highpass=False)
_, toy_signal_c = osc_signals(toy_slope, periodic_params=periodic_params_c,
                              highpass=False)

freq_a, toy_psd_a = sig.welch(toy_signal_a, **welch_params)
freq_b, toy_psd_b = sig.welch(toy_signal_b, **welch_params)
freq_c, toy_psd_c = sig.welch(toy_signal_c, **welch_params)

# Filter 1-100Hz
filt_a = (freq_a <= 100)
freq_a = freq_a[filt_a]
toy_psd_a = toy_psd_a[filt_a]
toy_psd_b = toy_psd_b[filt_a]
toy_psd_c = toy_psd_c[filt_a]

# %% Calc Aperiodic Component for largest range

freq_range = (lower_fitting_borders[0], upper_fitting_border)
freq0, psd_aperiodic_a, _, _ = irasa(toy_signal_a, band=freq_range, sf=srate)
_, psd_aperiodic_b, _, _ = irasa(toy_signal_b, band=freq_range, sf=srate)
_, psd_aperiodic_c, _, _ = irasa(toy_signal_c, band=freq_range, sf=srate)

psd_aperiodic_a = psd_aperiodic_a[0]
psd_aperiodic_b = psd_aperiodic_b[0]
psd_aperiodic_c = psd_aperiodic_c[0]
# %% a IRASA (takes very long)

fit_errors_a = calc_error(toy_signal_a)
fit_errors_b = calc_error(toy_signal_b)
fit_errors_c = calc_error(toy_signal_c)

error_plot_a = (lower_fitting_borders, fit_errors_a, c_error)
error_plot_b = (lower_fitting_borders, fit_errors_b, c_error)
error_plot_c = (lower_fitting_borders, fit_errors_c, c_error)


# %% Plot Params

width = 7.25  # inches
panel_fontsize = 12
legend_fontsize = 9
label_fontsize = 9
tick_fontsize = 9
annotation_fontsize = 7.5

mpl.rcParams['xtick.labelsize'] = tick_fontsize
mpl.rcParams['ytick.labelsize'] = tick_fontsize
mpl.rcParams['axes.labelsize'] = label_fontsize
mpl.rcParams['legend.fontsize'] = legend_fontsize
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


abc = dict(x=0, y=1.01, fontsize=panel_fontsize,
           fontdict=dict(fontweight="bold"))

# a
# a1
ymini = -13
ymaxi = -7
yticks_a1 = 10**np.arange(ymini, ymaxi, dtype=float)
ylim_a1 = (yticks_a1[0], yticks_a1[-1])
yticklabels_a1 = [""] * len(yticks_a1)
yticklabels_a1[0] = fr"$10^{{{ymini}}}$"
yticklabels_a1[-1] = fr"$10^{{{ymaxi}}}$"
ylabel_a1 = "PSD [a.u.]"

xticklabels_a1 = []
xlim_a = (1, 100)
axes_a1 = dict(xticklabels=xticklabels_a1, xlim=xlim_a, yticks=yticks_a1,
               yticklabels=yticklabels_a1, ylim=ylim_a1)
freqs123 = [freq1, freq2, freq3]
colors123 = [c_range1, c_range2, c_range3]
text_dic = dict(x=100, ha="right", fontsize=annotation_fontsize)

# a2
xticks_a2 = [1, 10, 100]
yticks_a2 = [0, .5, 1]
xlabel_a2 = "Lower fitting range border [Hz]"
ylabel_a2 = r"$|a_{truth} - a_{IRASA}|$"
ylim_a2 = (0, 1)
axes_a2 = dict(xticks=xticks_a2, xticklabels=xticks_a2, yticks=yticks_a2,
               xlim=xlim_a, xlabel=xlabel_a2, ylim=ylim_a2, ylabel=ylabel_a2)

axes_b1 = dict(xticklabels=xticklabels_a1, xlim=xlim_a, yticks=yticks_a1,
               yticklabels=[], ylim=ylim_a1)
axes_b2 = dict(xticks=xticks_a2, xticklabels=xticks_a2, yticks=yticks_a2,
               yticklabels=[],
               xlim=xlim_a, xlabel=xlabel_a2, ylim=ylim_a2)

# %% Plot
fig = plt.figure(figsize=[width, 5.2], constrained_layout=True)

gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1])

gs00 = gs0[0].subgridspec(2, 3)
axA1 = fig.add_subplot(gs00[0, 0])
axA2 = fig.add_subplot(gs00[1, 0])
axB1 = fig.add_subplot(gs00[0, 1])
axB2 = fig.add_subplot(gs00[1, 1])
axC1 = fig.add_subplot(gs00[0, 2])
axC2 = fig.add_subplot(gs00[1, 2])

gs01 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[1])
ax5 = fig.add_subplot(gs01[0])
ax6 = fig.add_subplot(gs01[1])
ax7 = fig.add_subplot(gs01[2])

# a
# a1
ax = axA1

# Plot sim
ax.loglog(freq_a, toy_psd_a, c_sim)
ax.loglog(freq0, psd_aperiodic_a, c_ap, zorder=0)

# Annotate fitting ranges
vline_dic = dict(ls="--", clip_on=False, alpha=.3)
ymin = ylim_a1[0]
for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
    y = toy_psd_a[freq_low]
    xmin = freq_low
    xmax = upper_fitting_border
    coords = (y, xmin, xmax)
    ax.hlines(*coords, color=color, ls="--")
    v_coords = (xmin, ymin, y)
    ax.vlines(*v_coords, color=color, **vline_dic)

    # Add annotation
    s = f"{freq_low}-{xmax}Hz"
    if i == 0:
        s = "Fitting range: " + s
        y = y**.97
    else:
        y = y**.98
    ax.text(s=s, y=y, **text_dic)

# Set axes
ax.text(s="a", **abc, transform=ax.transAxes)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.set(**axes_a1)
ax.set_ylabel(ylabel_a1, labelpad=-8)

# a2
ax = axA2

# Plot error
ax.semilogx(*error_plot_a)

# Annotate fitting ranges
for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
    xmin = freq_low
    ymin = 0
    ymax = 1.2
    v_coords = (xmin, ymin, ymax)
    ax.vlines(*v_coords, color=color, **vline_dic)

# Set axes
ax.set(**axes_a2)

# b
# b1
ax = axB1

# Plot sim
ax.loglog(freq_a, toy_psd_b, c_sim)
ax.loglog(freq0, psd_aperiodic_b, c_ap, zorder=0)

# Annotate fitting ranges
vline_dic = dict(ls="--", clip_on=False, alpha=.3)
ymin = ylim_a1[0]
for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
    y = toy_psd_b[freq_low]
    xmin = freq_low
    xmax = upper_fitting_border
    coords = (y, xmin, xmax)
    ax.hlines(*coords, color=color, ls="--")
    v_coords = (xmin, ymin, y)
    ax.vlines(*v_coords, color=color, **vline_dic)

    # Add annotation
    s = f"{freq_low}-{xmax}Hz"
    if i == 0:
        s = "Fitting range: " + s
        y = y**.97
    else:
        y = y**.98
    ax.text(s=s, y=y, **text_dic)

# Set axes
# ax.text(s="b", **abc, transform=ax.transAxes)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.set(**axes_b1)

# b2
ax = axB2

# Plot error
ax.semilogx(*error_plot_b)

# Annotate fitting ranges
for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
    xmin = freq_low
    ymin = 0
    ymax = 1.2
    v_coords = (xmin, ymin, ymax)
    ax.vlines(*v_coords, color=color, **vline_dic)

# Set axes
ax.set(**axes_b2)


# c
# c1
ax = axC1

# Plot sim
ax.loglog(freq_a, toy_psd_c, c_sim)
ax.loglog(freq0, psd_aperiodic_c, c_ap, zorder=0)

# Annotate fitting ranges
vline_dic = dict(ls="--", clip_on=False, alpha=.3)
ymin = ylim_a1[0]
for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
    y = toy_psd_c[freq_low]
    xmin = freq_low
    xmax = upper_fitting_border
    coords = (y, xmin, xmax)
    ax.hlines(*coords, color=color, ls="--")
    v_coords = (xmin, ymin, y)
    ax.vlines(*v_coords, color=color, **vline_dic)

    # Add annotation
    s = f"{freq_low}-{xmax}Hz"
    if i == 0:
        s = "Fitting range: " + s
        y = y**.97
    else:
        y = y**.98
    ax.text(s=s, y=y, **text_dic)

# Set axes
# ax.text(s="c", **abc, transform=ax.transAxes)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.set(**axes_b1)

# c2
ax = axC2

# Plot error
ax.semilogx(*error_plot_c)

# Annotate fitting ranges
for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
    xmin = freq_low
    ymin = 0
    ymax = 1.2
    v_coords = (xmin, ymin, ymax)
    ax.vlines(*v_coords, color=color, **vline_dic)

# Set axes
ax.set(**axes_b2)

plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()
