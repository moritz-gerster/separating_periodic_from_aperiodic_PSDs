"""Very broad peak widths require very large resampling factors."""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig
from pathlib import Path
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


def peak_width(freq, psd, ground_truth, freq_range=(1, 100), threshold=.001):
    """
    Calculate peak width as start- and endpoints from aperiodic ground truth.

    Parameters
    ----------
    freq : ndarray
        Freq array.
    psd : ndarray
        PSD Array including oscillation peaks.
    ground_truth : ndarray
        Aperiodic ground truth excluding oscillations peaks.
    freq_range : tuple of floats, optional
        Range to detect the peak. Multi-peak estimation is not supported.
        The default is (1, 100).
    threshold : float, optional
        Threshold for deviation from ground truth as percentage of peak
        maximum. The default is .001.

    Returns
    -------
    freq_start : float
        Frequency of peak start.
    freq_end : float
        Frequency of peak end.
    """
    # select range for difference calculation
    mask = (freq > freq_range[0]) & (freq < freq_range[1])
    psd_diff = np.abs(psd - ground_truth)[mask]

    # detect start and end indices
    arg_start = np.where(psd_diff > threshold * psd_diff.max())[0][0]
    arg_end = np.where(psd_diff > threshold * psd_diff.max())[0][-1]

    # add lower freq range border to correct index
    arg_start += (freq <= freq_range[0]).sum()
    arg_end += (freq <= freq_range[0]).sum()

    return freq[arg_start], freq[arg_end]


# %% PARAMETERS

# Signal params
srate = 2400
nperseg = srate
welch_params = dict(fs=srate, nperseg=nperseg)

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig5_IRASA_PeakWidth.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# File names
path = "../data/Fig5/"

# Colors
# a)
c_sim = "k"
c_error = "r"

c_range1 = "b"
c_range2 = "g"
c_range3 = "y"

c_ap = "grey"

# b)
c_IRASA1 = "c"
c_IRASA2 = "m"
c_IRASA3 = "y"

colors123 = [c_range1, c_range2, c_range3]
# %% a Sim Toy Signal with Three Oscillations and Fit

# fit in all frequency ranges from 1 to 80...
lower_fitting_borders = range(1, 80)
upper_fitting_border = 100  # ... to 100 Hz

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

periodic_params_c = [(freq1, amp*4.4, width*.8),
                     (freq2, amp*1.7, width*1.2),
                     (freq3, amp*.7, width*2.5)]
# Sim Toy Signal
toy_ground_a, toy_signal_a = osc_signals(toy_slope,
                                         periodic_params=periodic_params_a,
                                         highpass=False)
toy_ground_b, toy_signal_b = osc_signals(toy_slope,
                                         periodic_params=periodic_params_b,
                                         highpass=False)
toy_ground_c, toy_signal_c = osc_signals(toy_slope,
                                         periodic_params=periodic_params_c,
                                         highpass=False)

freq_a, toy_psd_a = sig.welch(toy_signal_a, **welch_params)
_, toy_psd_b = sig.welch(toy_signal_b, **welch_params)
_, toy_psd_c = sig.welch(toy_signal_c, **welch_params)

_, ground_psd_a = sig.welch(toy_ground_a, **welch_params)
_, ground_psd_b = sig.welch(toy_ground_b, **welch_params)
_, ground_psd_c = sig.welch(toy_ground_c, **welch_params)

# Filter 1-100Hz
filt_a = (freq_a <= 100)
freq_a = freq_a[filt_a]

toy_psd_a = toy_psd_a[filt_a]
toy_psd_b = toy_psd_b[filt_a]
toy_psd_c = toy_psd_c[filt_a]

ground_psd_a = ground_psd_a[filt_a]
ground_psd_b = ground_psd_b[filt_a]
ground_psd_c = ground_psd_c[filt_a]

freqs123 = [freq1, freq2, freq3]

# %% A Calc peak widths
range1 = (1, 10)
range2 = (10, 20)
range3 = (20, 100)

# Start and end points xaxis
xmin_a11, xmax_a11 = peak_width(freq_a, toy_psd_a, ground_psd_a,
                                freq_range=range1)
xmin_a12, xmax_a12 = peak_width(freq_a, toy_psd_a, ground_psd_a,
                                freq_range=range2)
xmin_a13, xmax_a13 = peak_width(freq_a, toy_psd_a, ground_psd_a,
                                freq_range=range3)

xmin_a21, xmax_a21 = peak_width(freq_a, toy_psd_b, ground_psd_b,
                                freq_range=range1)
xmin_a22, xmax_a22 = peak_width(freq_a, toy_psd_b, ground_psd_b,
                                freq_range=range2)
xmin_a23, xmax_a23 = peak_width(freq_a, toy_psd_b, ground_psd_b,
                                freq_range=range3)

xmin_a31, xmax_a31 = peak_width(freq_a, toy_psd_c, ground_psd_c,
                                freq_range=range1)
xmin_a32, xmax_a32 = peak_width(freq_a, toy_psd_c, ground_psd_c,
                                freq_range=range2)
xmin_a33, xmax_a33 = peak_width(freq_a, toy_psd_c, ground_psd_c,
                                freq_range=range3)

# freq resolution too low for proper estimation of first peak:
xmin_a11 -= 1
xmax_a11 += 1

# Start and end points yaxis
ylow_a11 = toy_psd_a[np.argmin(np.abs(freq_a - xmin_a11))]
yhigh_a11 = toy_psd_a[np.argmin(np.abs(freq_a - xmax_a11))]
ylow_a12 = toy_psd_a[np.argmin(np.abs(freq_a - xmin_a12))]
yhigh_a12 = toy_psd_a[np.argmin(np.abs(freq_a - xmax_a12))]
ylow_a13 = toy_psd_a[np.argmin(np.abs(freq_a - xmin_a13))]
yhigh_a13 = toy_psd_a[np.argmin(np.abs(freq_a - xmax_a13))]

ylow_a21 = toy_psd_b[np.argmin(np.abs(freq_a - xmin_a21))]
yhigh_a21 = toy_psd_b[np.argmin(np.abs(freq_a - xmax_a21))]
ylow_a22 = toy_psd_b[np.argmin(np.abs(freq_a - xmin_a22))]
yhigh_a22 = toy_psd_b[np.argmin(np.abs(freq_a - xmax_a22))]
ylow_a23 = toy_psd_b[np.argmin(np.abs(freq_a - xmin_a23))]
yhigh_a23 = toy_psd_b[np.argmin(np.abs(freq_a - xmax_a23))]

ylow_a31 = toy_psd_c[np.argmin(np.abs(freq_a - xmin_a31))]
yhigh_a31 = toy_psd_c[np.argmin(np.abs(freq_a - xmax_a31))]
ylow_a32 = toy_psd_c[np.argmin(np.abs(freq_a - xmin_a32))]
yhigh_a32 = toy_psd_c[np.argmin(np.abs(freq_a - xmax_a32))]
ylow_a33 = toy_psd_c[np.argmin(np.abs(freq_a - xmin_a33))]
yhigh_a33 = toy_psd_c[np.argmin(np.abs(freq_a - xmax_a33))]
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


# %% B Sim
h_max_s = 2
h_max_m = 8
h_max_l = 35

band = (1, 599)

srate_small = band[1] * 2 * h_max_s  # avoid nyquist frequency
srate_med = band[1] * 2 * h_max_m
srate_large = band[1] * 2 * h_max_l

welch_params_small = dict(fs=srate_small, nperseg=4*srate_small)
welch_params_med = dict(fs=srate_med, nperseg=4*srate_med)
welch_params_large = dict(fs=srate_large, nperseg=4*srate_large)


# Oscillations parameters:
toy_slope = 2

center_freq = 30
amp_s = 1.1
amp_m = 2.1
amp_l = 3.2

width_s = amp_s
width_m = amp_m
width_l = amp_l

periodic_params_s = [(center_freq, amp_s, width_s),
                     (10*center_freq, amp_s, 10*width_s)]
periodic_params_m = [(center_freq, amp_m, width_m),
                     (10*center_freq, amp_m, 10*width_m)]
periodic_params_l = [(center_freq, amp_l, width_l),
                     (10*center_freq, amp_l, 10*width_l)]

# Sim Toy Signal
noise_small, peak_small = osc_signals(toy_slope,
                                      periodic_params=periodic_params_s,
                                      highpass=False, srate=srate_small)
noise_med, peak_med = osc_signals(toy_slope,
                                  periodic_params=periodic_params_m,
                                  highpass=False, srate=srate_med)
noise_large, peak_large = osc_signals(toy_slope,
                                      periodic_params=periodic_params_l,
                                      highpass=False, srate=srate_large)

freq_small, peak_psd_small = sig.welch(peak_small, **welch_params_small)
freq_med, peak_psd_med = sig.welch(peak_med, **welch_params_med)
freq_large, peak_psd_large = sig.welch(peak_large, **welch_params_large)

_, noise_psd_small = sig.welch(noise_small, **welch_params_small)
_, noise_psd_med = sig.welch(noise_med, **welch_params_med)
_, noise_psd_large = sig.welch(noise_large, **welch_params_large)

# Filter 1-100Hz
filt_small = (freq_small <= 1000)
filt_med = (freq_med <= 1000)
filt_large = (freq_large <= 1000)

freq_small = freq_small[filt_small]
freq_med = freq_med[filt_med]
freq_large = freq_large[filt_large]

peak_psd_small = peak_psd_small[filt_small]
peak_psd_med = peak_psd_med[filt_med]
peak_psd_large = peak_psd_large[filt_large]

noise_psd_small = noise_psd_small[filt_small]
noise_psd_med = noise_psd_med[filt_med]
noise_psd_large = noise_psd_large[filt_large]


# %% B Calc IRASA

N_h = 16

irasa_params_small = dict(band=band, hset=np.linspace(1.1, h_max_s, N_h))
irasa_params_med = dict(band=band, hset=np.linspace(1.1, h_max_m, N_h))
irasa_params_large = dict(band=band, hset=np.linspace(1.1, h_max_l, N_h))

IRASA_sim_small_h1 = irasa(peak_small, sf=srate_small, **irasa_params_small)
print("IRASA small done")

IRASA_sim_med_h1 = irasa(peak_med, sf=srate_med, **irasa_params_small)
print("IRASA medium 1/2 done")
IRASA_sim_med_h2 = irasa(peak_med, sf=srate_med, **irasa_params_med)
print("IRASA medium 2/2 done")

IRASA_sim_large_h1 = irasa(peak_large, sf=srate_large, **irasa_params_small)
print("IRASA large 1/3 done")
IRASA_sim_large_h2 = irasa(peak_large, sf=srate_large, **irasa_params_med)
print("IRASA large 2/3 done")
IRASA_sim_large_h3 = irasa(peak_large, sf=srate_large, **irasa_params_large)
print("IRASA large 3/3 done")

freqs_sim_s, ap_sim_small_h1, per1_s, params1_s = IRASA_sim_small_h1

freqs_sim_m, ap_sim_med_h1, per1_m, params1_m = IRASA_sim_med_h1
freqs_sim_m, ap_sim_med_h2, per2_m, params2_m = IRASA_sim_med_h2

freqs_sim_l, ap_sim_large_h1, per1_l, params1_l = IRASA_sim_large_h1
freqs_sim_l, ap_sim_large_h2, per2_l, params2_l = IRASA_sim_large_h2
freqs_sim_l, ap_sim_large_h3, per3_l, params3_l = IRASA_sim_large_h3

# Normalize
# Divide repetitive plots by 100 or 10000 to avoid overlap
ap_sim_small_h1 = ap_sim_small_h1[0] / peak_psd_small[0]
ap_sim_med_h1 = ap_sim_med_h1[0] / peak_psd_med[0]
ap_sim_med_h2 = ap_sim_med_h2[0] / peak_psd_med[0] / 10000
ap_sim_large_h1 = ap_sim_large_h1[0] / peak_psd_large[0]
ap_sim_large_h2 = ap_sim_large_h2[0] / peak_psd_large[0] / 100
ap_sim_large_h3 = ap_sim_large_h3[0] / peak_psd_large[0] / 10000

peak_psd_small /= peak_psd_small[0]
peak_psd_med /= peak_psd_med[0]
peak_psd_large /= peak_psd_large[0]

noise_psd_small /= noise_psd_small[0]
noise_psd_med /= noise_psd_med[0]
noise_psd_large /= noise_psd_large[0]

peak_psd_med_low = peak_psd_med / 10000

peak_psd_large_low = peak_psd_large / 100
peak_psd_large_lower = peak_psd_large / 10000


# %% B Determine peak width

frange1 = (10, 100)
frange2 = (100, 1000)

# Peak start and endpoints xaxis
xmin_small_min1, xmin_small_max1 = peak_width(freq_small,
                                              peak_psd_small,
                                              noise_psd_small,
                                              freq_range=frange1)
xmin_small_min2, xmin_small_max2 = peak_width(freq_small,
                                              peak_psd_small,
                                              noise_psd_small,
                                              freq_range=frange2)
xmin_med_min1, xmin_med_max1 = peak_width(freq_med, peak_psd_med,
                                          noise_psd_med,
                                          freq_range=frange1)
xmin_med_min2, xmin_med_max2 = peak_width(freq_med, peak_psd_med,
                                          noise_psd_med,
                                          freq_range=frange2)
xmin_large_min1, xmin_large_max1 = peak_width(freq_large,
                                              peak_psd_large,
                                              noise_psd_large,
                                              freq_range=frange1)
xmin_large_min2, xmin_large_max2 = peak_width(freq_large,
                                              peak_psd_large,
                                              noise_psd_large,
                                              freq_range=frange2)
# Peak start and endpoints yaxis
ylow_small1 = peak_psd_small[np.argmin(np.abs(freq_small - xmin_small_min1))]
yhigh_small1 = peak_psd_small[np.argmin(np.abs(freq_small - xmin_small_max1))]
ylow_small2 = peak_psd_small[np.argmin(np.abs(freq_small - xmin_small_min2))]
yhigh_small2 = peak_psd_small[np.argmin(np.abs(freq_small - xmin_small_max2))]

ylow_med1 = peak_psd_med_low[np.argmin(np.abs(freq_med - xmin_med_min1))]
yhigh_med1 = peak_psd_med_low[np.argmin(np.abs(freq_med - xmin_med_max1))]
ylow_med2 = peak_psd_med_low[np.argmin(np.abs(freq_med - xmin_med_min2))]
yhigh_med2 = peak_psd_med_low[np.argmin(np.abs(freq_med - xmin_med_max2))]

ylow_large1 = peak_psd_large_lower[np.argmin(np.abs(freq_large
                                                    - xmin_large_min1))]
yhigh_large1 = peak_psd_large_lower[np.argmin(np.abs(freq_large
                                                     - xmin_large_max1))]
ylow_large2 = peak_psd_large_lower[np.argmin(np.abs(freq_large
                                                    - xmin_large_min2))]
yhigh_large2 = peak_psd_large_lower[np.argmin(np.abs(freq_large
                                                     - xmin_large_max2))]

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


panel_labels = dict(x=0, y=1.01, fontsize=panel_fontsize,
                    fontdict=dict(fontweight="bold"))

# a11
xticklabels_a1 = []
xlim_a = (1, 100)

ymini, ymaxi = (-13, -7)
yticks_a1 = 10**np.arange(ymini, ymaxi, dtype=float)
ylim_a1 = (yticks_a1[0], yticks_a1[-1])
yticklabels_a1 = [""] * len(yticks_a1)
yticklabels_a1[0] = fr"$10^{{{ymini}}}$"
yticklabels_a1[-1] = fr"$10^{{{ymaxi}}}$"
ylabel_a1 = "PSD [a.u.]"

axes_a1 = dict(xticklabels=xticklabels_a1, xlim=xlim_a, yticks=yticks_a1,
               yticklabels=yticklabels_a1, ylim=ylim_a1)

# a12
xlabel_a2 = "Lower fitting range border [Hz]"
xticks_a2 = [1, 10, 100]

ylabel_a2 = r"$|\beta_{truth} - \beta_{IRASA}|$"
yticks_a2 = [0, .5, 1]
ylim_a2 = (0, 1)

axes_a2 = dict(xticks=xticks_a2, xticklabels=xticks_a2, yticks=yticks_a2,
               xlim=xlim_a, ylim=ylim_a2, ylabel=ylabel_a2)
axes_a21 = dict(xticklabels=xticklabels_a1, xlim=xlim_a, yticks=yticks_a1,
                yticklabels=[], ylim=ylim_a1)
axes_a22 = dict(xticks=xticks_a2, xticklabels=xticks_a2, yticks=yticks_a2,
                yticklabels=[],
                xlim=xlim_a, ylim=ylim_a2)

# Fit ranges
height_a = ylim_a1[0] * 2
fit_range_a11 = dict(xmin=xmin_a11, xmax=xmax_a11,
                     ylow=ylow_a11, yhigh=yhigh_a11,
                     height=height_a, annotate_pos="below", annotation="diff")
fit_range_a12 = dict(xmin=xmin_a12, xmax=xmax_a12,
                     ylow=ylow_a12, yhigh=yhigh_a12,
                     height=height_a, annotate_pos="below", annotation="diff")
fit_range_a13 = dict(xmin=xmin_a13, xmax=xmax_a13,
                     ylow=ylow_a13, yhigh=yhigh_a13,
                     height=height_a, annotate_pos="below", annotation="diff")

fit_range_a21 = dict(xmin=xmin_a21, xmax=xmax_a21,
                     ylow=ylow_a21, yhigh=yhigh_a21,
                     height=height_a, annotate_pos="below", annotation="diff")
fit_range_a22 = dict(xmin=xmin_a22, xmax=xmax_a22,
                     ylow=ylow_a22, yhigh=yhigh_a22,
                     height=height_a, annotate_pos="below", annotation="diff")
fit_range_a23 = dict(xmin=xmin_a23, xmax=xmax_a23,
                     ylow=ylow_a23, yhigh=yhigh_a23,
                     height=height_a, annotate_pos="below", annotation="diff")

fit_range_a31 = dict(xmin=xmin_a31, xmax=xmax_a31,
                     ylow=ylow_a31, yhigh=yhigh_a31,
                     height=height_a, annotate_pos="below", annotation="diff")
fit_range_a32 = dict(xmin=xmin_a32, xmax=xmax_a32,
                     ylow=ylow_a32, yhigh=yhigh_a32,
                     height=height_a, annotate_pos="below", annotation="diff")
fit_range_a33 = dict(xmin=xmin_a33, xmax=xmax_a33,
                     ylow=ylow_a33, yhigh=yhigh_a33,
                     height=height_a, annotate_pos="below", annotation="diff")

# b
xlabel_b = "Frequency [Hz]"
xlim_b = (.7, 1000)
xticks_b = [1, 10, 100, 600]

ymini, ymaxi = (-10, 1)
yticks_b = 10**np.arange(ymini, ymaxi, dtype=float)
yticklabels_b = [r"$10^{-10}$", "", "", "", "", "",
                 "", "", "", "", r"$10^0$"]
ylim_b = (5e-11, 1.1)

axes_b1 = dict(xticks=xticks_b, xticklabels=xticks_b, yticks=yticks_b,
               yticklabels=yticklabels_b, xlim=xlim_b, ylim=ylim_b,
               xlabel=xlabel_b)

axes_b2 = dict(xticks=xticks_b, xticklabels=xticks_b, yticks=yticks_b,
               yticklabels=[], xlim=xlim_b, ylim=ylim_b, xlabel=xlabel_b)

# Annotations
height_b = ylim_b[0] * 4
anno_small1 = dict(xmin=xmin_small_min1, xmax=xmin_small_max1,
                   ylow=ylow_small1, yhigh=yhigh_small1,
                   height=height_b, annotate_pos="left", annotation="log-diff")
anno_small2 = dict(xmin=xmin_small_min2, xmax=xmin_small_max2,
                   ylow=ylow_small2, yhigh=yhigh_small2,
                   height=height_b, annotate_pos="left", annotation="log-diff")

anno_med1 = dict(xmin=xmin_med_min1, xmax=xmin_med_max1,
                 ylow=ylow_med1, yhigh=yhigh_med1,
                 height=height_b, annotate_pos="left", annotation="log-diff")
anno_med2 = dict(xmin=xmin_med_min2, xmax=xmin_med_max2,
                 ylow=ylow_med2, yhigh=yhigh_med2,
                 height=height_b, annotate_pos="left", annotation="log-diff")

anno_large1 = dict(xmin=xmin_large_min1, xmax=xmin_large_max1,
                   ylow=ylow_large1, yhigh=yhigh_large1,
                   height=height_b, annotate_pos="left", annotation="log-diff")
anno_large2 = dict(xmin=xmin_large_min2, xmax=xmin_large_max2,
                   ylow=ylow_large2, yhigh=yhigh_large2,
                   height=height_b, annotate_pos="left", annotation="log-diff")

leg = dict(labelspacing=.1, loc=1, borderaxespad=0, borderpad=.1,
           handletextpad=.2, handlelength=2)


def draw_fitrange(ax1, ax2, toy_psd, freqs, colors):
    """Make horizontal lines to annotate fit range."""
    text_dic = dict(x=100, ha="right", fontsize=annotation_fontsize)
    vline_dic = dict(ls="--", clip_on=False, alpha=.3)
    for i, (freq_low, color) in enumerate(zip(freqs, colors)):
        ymin = ylim_a1[0]
        y = toy_psd[freq_low]
        xmin = freq_low
        xmax = upper_fitting_border
        coords = (y, xmin, xmax)
        ax1.hlines(*coords, color=color, ls="--")
        v_coords = (xmin, ymin, y)
        ax1.vlines(*v_coords, color=color, **vline_dic)

        # Add annotation
        s = f"{freq_low}-{xmax}Hz"
        if i == 0:
            s = "Fitting range: " + s
            y = y**.97
        else:
            y = y**.98
        ax1.text(s=s, y=y, **text_dic)
        # Add vlines below
        ymin = 0
        ymax = 1.4
        v_coords = (xmin, ymin, ymax)
        ax2.vlines(*v_coords, color=color, **vline_dic)


def annotate_fit_range(ax, xmin, xmax, height, ylow=None, yhigh=None,
                       annotate_pos=True, annotation="log-diff"):
    """
    Annotate fitting range.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Ax to draw the lines.
    xmin : float
        X-range minimum.
    xmax : float
        X-range maximum.
    ylow : float, optional
        Position on y-axis to connect the vertical lines. If None, no vertical
        lines are drawn. The default is None.
    yhigh : float, optional
        Position on y-axis to connect the vertical lines. If None, no vertical
        lines are drawn. The default is None.
    height : float
        Position on y-axis of range.
    annotate_middle : bool, optional
        Whether to annotate the frequency in the middle of the range (prettier)
        or if it is too small next to the range. The default is True.
    annotate_range : bool, optional
        Whether to annotate the range as min-max or the difference as max-min.
        By default, the range min-max is annotated.

    Returns
    -------
    None.

    """
    text_pos = 10**((np.log10(xmin) + np.log10(xmax)) / 2)
    box_alpha = 1
    ha = "center"
    text_height = height
    if annotate_pos == "below":
        text_height = 1.5e-1 * height
        box_alpha = 0
    elif isinstance(annotate_pos, (int, float)):
        text_height *= annotate_pos
        box_alpha = 0
    elif annotate_pos == "left":
        box_alpha = 0
        ha = "right"
        text_pos = xmin * .9

    # Plot Values
    arrow_dic = dict(text="", xy=(xmin, height), xytext=(xmax, height),
                     arrowprops=dict(arrowstyle="|-|, widthA=.3, widthB=.3",
                                     shrinkA=0, shrinkB=0))
    anno_dic = dict(ha=ha, va="center", bbox=dict(fc="white", ec="none",
                    boxstyle="square,pad=0.2", alpha=box_alpha),
                    fontsize=annotation_fontsize)
    vline_dic = dict(color="k", lw=.5, ls=":")

    ax.annotate(**arrow_dic)

    if annotation == "diff":
        range_str = f"{xmax-xmin:.0f}Hz"
    elif annotation == "log-diff":
        range_str = (f"{xmax-xmin:.0f}Hz"
                     f"\n{(np.log10(xmax)-np.log10(xmin)):.2f}")
        # replace by np.log10(xmax/xmin)
    else:
        range_str = f"{xmin:.1f}-{xmax:.1f}Hz"
    ax.text(text_pos, text_height, s=range_str, **anno_dic)

    if ylow and yhigh:
        ax.vlines(xmin, height, ylow, **vline_dic)
        ax.vlines(xmax, height, yhigh, **vline_dic)


# %% Plot

fig, ax = plt.subplots(3, 3, figsize=[width, 5.5],
                       constrained_layout=True,
                       gridspec_kw=dict(height_ratios=[5, 4, 10]))

ax_a11 = ax[0, 0]
ax_a12 = ax[1, 0]
ax_a21 = ax[0, 1]
ax_a22 = ax[1, 1]
ax_a31 = ax[0, 2]
ax_a32 = ax[1, 2]

ax_b1 = ax[2, 0]
ax_b2 = ax[2, 1]
ax_b3 = ax[2, 2]

# a
# a11
ax = ax_a11

# Plot sim
ax.loglog(freq_a, toy_psd_a, c_sim)
ax.loglog(freq0, psd_aperiodic_a, c_ap, zorder=0)
draw_fitrange(ax_a11, ax_a12, toy_psd_a, freqs123, colors123)

annotate_fit_range(ax, **fit_range_a11)
annotate_fit_range(ax, **fit_range_a12)
annotate_fit_range(ax, **fit_range_a13)

# Set axes
ax.text(s="a", **panel_labels, transform=ax.transAxes)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.set(**axes_a1)
ax.set_ylabel(ylabel_a1, labelpad=-5)

# a12
ax = ax_a12
ax.semilogx(*error_plot_a)

# Set axes
ax.set(**axes_a2)

# a21
ax = ax_a21

# Plot sim
ax.loglog(freq_a, toy_psd_b, c_sim)
ax.loglog(freq0, psd_aperiodic_b, c_ap, zorder=0)
draw_fitrange(ax_a21, ax_a22, toy_psd_b, freqs123, colors123)

annotate_fit_range(ax, **fit_range_a21)
annotate_fit_range(ax, **fit_range_a22)
annotate_fit_range(ax, **fit_range_a23)

# Set axes
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.set(**axes_a21)

# a22
ax = ax_a22
ax.semilogx(*error_plot_b)
ax.set_xlabel(xlabel_a2)

# Set axes
ax.set(**axes_a22)


# a31
ax = ax_a31
ax.loglog(freq_a, toy_psd_c, c_sim)
ax.loglog(freq0, psd_aperiodic_c, c_ap, zorder=0)
draw_fitrange(ax_a31, ax_a32, toy_psd_c, freqs123, colors123)

annotate_fit_range(ax, **fit_range_a31)
annotate_fit_range(ax, **fit_range_a32)
annotate_fit_range(ax, **fit_range_a33)

# Set axes
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.set(**axes_a21)

# a32
ax = ax_a32
ax.semilogx(*error_plot_c)

# Set axes
ax.set(**axes_a22)


# b1
ax = ax_b1
ax.loglog(freq_small, peak_psd_small, c_sim)
ax.loglog(freqs_sim_s, ap_sim_small_h1, c_IRASA1,
          label=r"$h_{max}$="f"{h_max_s}")

# annotate freq bandwidth
annotate_fit_range(ax, **anno_small1)
annotate_fit_range(ax, **anno_small2)

# Set axes
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=20)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.set_ylabel(ylabel_a1, labelpad=-5)

ax.set(**axes_b1)
ax.text(s="b", **panel_labels, transform=ax.transAxes)
ax.legend(**leg)


# b2
ax = ax_b2
ax.loglog(freq_med, peak_psd_med, c_sim)
ax.loglog(freqs_sim_m, ap_sim_med_h1, c_IRASA1)

ax.loglog(freq_med, peak_psd_med_low, c_sim, alpha=.5)
ax.loglog(freqs_sim_m, ap_sim_med_h2, c_IRASA2,
          label=r"$h_{max}$="f"{h_max_m}")

# annotate freq bandwidth
annotate_fit_range(ax, **anno_med1)
annotate_fit_range(ax, **anno_med2)

# Set axes
ax.set(**axes_b2)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=20)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.legend(**leg)


# b3
ax = ax_b3
ax.loglog(freq_large, peak_psd_large, c_sim)
ax.loglog(freqs_sim_l, ap_sim_large_h1, c_IRASA1)

ax.loglog(freq_large, peak_psd_large_low, c_sim, alpha=.5)
ax.loglog(freqs_sim_l, ap_sim_large_h2, c_IRASA2)

ax.loglog(freq_large, peak_psd_large_lower, c_sim, alpha=.5)
ax.loglog(freqs_sim_l, ap_sim_large_h3, c_IRASA3,
          label=r"$h_{max}$="f"{h_max_l}")

# annotate freq bandwidth
annotate_fit_range(ax, **anno_large1)
annotate_fit_range(ax, **anno_large2)

# Set axes
ax.set(**axes_b2)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=20)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.legend(**leg)


plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()
