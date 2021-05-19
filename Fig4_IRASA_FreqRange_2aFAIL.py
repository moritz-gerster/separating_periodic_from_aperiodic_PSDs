"""
The evaluated frequency range is larger than the fitting range.

Panel a: Same as panel a Fig 2.

"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig
# import mne
from pathlib import Path
from fooof import FOOOF
# from fooof.sim.gen import gen_aperiodic
import matplotlib.gridspec as gridspec
from noise_helper import irasa


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


# %% PARAMETERS

# Signal params
srate = 2400
nperseg = srate  # 4*srate too high resolution for fooof
welch_params = dict(fs=srate, nperseg=nperseg)

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig3_IRASA_FreqRange.pdf"
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
width1, width2, width3 = .01, .01, .01
toy_slope = 1
nlv = 0.000001
nlv = 0.00001

periodic_params = [(freq1, amp1, width1),
                   (freq2, amp2, width2),
                   (freq3, amp3, width3)]

# Sim Toy Signal
_, toy_signal = osc_signals(toy_slope, periodic_params=periodic_params,
                            nlv=nlv)
freq, toy_psd = sig.welch(toy_signal, **welch_params)

# Filter 1-100Hz
filt = (freq > 0) & (freq <= 100)
freq = freq[filt]
toy_psd = toy_psd[filt]

toy_plot = (freq, toy_psd, c_sim)

# %% IRASA
# hset_max = srate / 4 / freq_range[1]
win_sec = 4
irasa_params = {"sf": srate, "win_sec": win_sec}

lower_fitting_borders = np.arange(1, 69, 2)

fit_errors_IRASA = []
for low in lower_fitting_borders:
    freq_range = (low, low+32)
    IRASA = irasa(data=toy_signal, band=freq_range, **irasa_params)
    _, _, _, params = IRASA
    exp = -params["Slope"][0]
    error = np.abs(toy_slope - exp)
    fit_errors_IRASA.append(error)
    fit_errors_IRASA.append(error)


# %% Fooof

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

# %% Plot

error_plot = (np.arange(1, 69, 1), fit_errors_IRASA, c_error)

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
