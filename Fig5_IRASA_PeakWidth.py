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
import mne
from fooof import FOOOF
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
c_IRASA1 = "C1"
c_IRASA2 = "C2"
c_IRASA3 = "orangered"

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

periodic_params_c = [(freq1, amp*4.4, width*.8),
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


# %% C Real data

srate = 2400
nperseg = srate
welch_params = dict(fs=srate, nperseg=nperseg)

fname_MEG = "Subj016_ON_1_raw.fif"
fname_LFP = "Subj016_OFF_001_STN_r+s_raw.fif"

sub_MEG = mne.io.read_raw_fif(path + fname_MEG, preload=True)
sub_LFP = mne.io.read_raw_fif(path + fname_LFP, preload=True)

small_grad = "MEG0913"
med_mag = "MEG2421"
large_LFP = "R2-R3"

channels = [small_grad, med_mag]

sub_MEG.pick_channels(channels)
sub_LFP.pick_channels([large_LFP])

# Convert to numpy
MEG_raw = sub_MEG.get_data()
LFP_raw = sub_LFP.get_data()[0]
LFP_raw *= 1e6  # convert V to uV


freq, spec_MEG = sig.welch(MEG_raw, **welch_params)
freq, spec_LFP = sig.welch(LFP_raw, **welch_params)


filt = (freq >= 1) & (freq <= 600)
freq = freq[filt]
spec_MEG = spec_MEG[:, filt]
spec_LFP = spec_LFP[filt]


plot_small = (freq, spec_MEG[0], c_real)
plot_med = (freq, spec_MEG[1], c_real)
plot_large = (freq, spec_LFP, c_real)

# %% C Apply fooof to determine peak width

real_s = FOOOF(max_n_peaks=2, verbose=False)
real_m = FOOOF(max_n_peaks=1, verbose=False, peak_width_limits=(0.5, 150))
real_m12 = FOOOF(max_n_peaks=2, verbose=False, peak_width_limits=(0.5, 150))
real_l = FOOOF(max_n_peaks=1, verbose=False, peak_width_limits=(.5, 150))

real_s.fit(freq, spec_MEG[0], (3, 20))
real_m.fit(freq, spec_MEG[1], (1, 100))
real_m12.fit(freq, spec_MEG[1], (1, 100))
real_l.fit(freq, spec_LFP, (10, 150))

real_l.plot(plt_log=True)

freq_real_s, pow_real_s, bw_real_s = real_s.peak_params_[0]
freq_real_m, pow_real_m, bw_real_m = real_m.peak_params_[0]
freq_real_m1, pow_real_m1, bw_real_m1 = real_m12.peak_params_[0]
freq_real_m2, pow_real_m2, bw_real_m2 = real_m12.peak_params_[1]
freq_real_l, pow_real_l, bw_real_l = real_l.peak_params_[0]

# %% C Calc IRASA
h_max1 = 2
band = (2, 100)
N_h = 16
irasa_params = dict(sf=srate, band=band, hset=np.linspace(1.1, h_max1, N_h))

IRASA_small_h1 = irasa(MEG_raw[0], **irasa_params)
IRASA_med_h1 = irasa(MEG_raw[1], **irasa_params)
IRASA_large_h1 = irasa(LFP_raw, **irasa_params)

freq_I, ap_small19, per_small19, params_small19 = IRASA_small_h1
_, ap_med19, per_med19, params_med19 = IRASA_med_h1
_, ap_large19, per_large19, params_large19 = IRASA_large_h1

plot_ap_small_h1 = (freq_I, ap_small19[0], c_IRASA1)
plot_ap_med19 = (freq_I, ap_med19[0], c_IRASA1)
plot_ap_large19 = (freq_I, ap_large19[0], c_IRASA1)


h_max = 4.9
irasa_params["hset"] = np.linspace(1.1, h_max, N_h)
#IRASA_s49 = irasa(MEG_raw[0], **irasa_params)
IRASA_m49 = irasa(MEG_raw[1], **irasa_params)
IRASA_l49 = irasa(LFP_raw, **irasa_params)

#freq_I, ap_small49, per_small49, params_small49 = IRASA_s49
_, ap_med49, per_med49, params_med49 = IRASA_m49
_, ap_large49, per_large49, params_large49 = IRASA_l49

#plot_ap_small49 = (freq_I, ap_small49[0], c_IRASA2)
plot_ap_med49 = (freq_I, ap_med49[0], c_IRASA2)
plot_ap_large49 = (freq_I, ap_large49[0], c_IRASA2)


# %% B Sim

h_max2 = 20
h_max3 = 45

srate_sim = band[1] * 2 * h_max3  # avoid nyquist frequency
nperseg = srate_sim
welch_params = dict(fs=srate_sim, nperseg=nperseg)


# Oscillations parameters:
toy_slope = 2

periodic_params_s = [(freq_real_s, pow_real_s, bw_real_s/4)]
periodic_params_m = [(freq_real_m, pow_real_m*8, bw_real_m/3)]
periodic_params_m12 = [(freq_real_m1, pow_real_m1, bw_real_m1/4),
                       (freq_real_m2, pow_real_m2, bw_real_m2/4)]
periodic_params_l = [(freq_real_l, pow_real_l*10, bw_real_l/2)]

# Sim Toy Signal
_, peak_small = osc_signals(toy_slope, periodic_params=periodic_params_s,
                            highpass=False, srate=srate_sim)
_, peak_med = osc_signals(toy_slope, periodic_params=periodic_params_m,
                          highpass=False, srate=srate_sim)
_, peak_med12 = osc_signals(toy_slope, periodic_params=periodic_params_m12,
                          highpass=False, srate=srate_sim)
_, peak_large = osc_signals(toy_slope, periodic_params=periodic_params_l,
                            highpass=False, srate=srate_sim)

freq_b, peak_psd_small = sig.welch(peak_small, **welch_params)
freq_b, peak_psd_med = sig.welch(peak_med, **welch_params)
freq_b, peak_psd_med12 = sig.welch(peak_med, **welch_params)
freq_b, peak_psd_large = sig.welch(peak_large, **welch_params)

# Filter 1-100Hz
filt_b = (freq_b >= 1) & (freq_b <= 600)
freq_b = freq_b[filt_b]
peak_psd_small = peak_psd_small[filt_b]
peak_psd_med = peak_psd_med[filt_b]
peak_psd_med12 = peak_psd_med12[filt_b]
peak_psd_large = peak_psd_large[filt_b]

plot_psd_small = (freq_b, peak_psd_small, c_sim)
plot_psd_med = (freq_b, peak_psd_med, c_sim)
plot_psd_med12 = (freq_b, peak_psd_med12, c_sim)
plot_psd_large = (freq_b, peak_psd_large, c_sim)

# %% B Apply fooof to determine peak width

sim_s = FOOOF(max_n_peaks=1, verbose=False)
sim_m = FOOOF(max_n_peaks=1, verbose=False)
sim_m12 = FOOOF(max_n_peaks=2, verbose=False)
sim_l = FOOOF(max_n_peaks=1, verbose=False)

sim_s.fit(freq_b, peak_psd_small, (3, 20))
sim_m.fit(freq_b, peak_psd_med, (1, 200))
sim_m12.fit(freq_b, peak_psd_med, (1, 200))
sim_l.fit(freq_b, peak_psd_large, (10, 150))

# sim_m12.plot(plt_log=True)

bw_sim_s = sim_s.peak_params_[0, 2]
bw_sim_m = sim_m.peak_params_[0, 2]
bw_sim_m12 = sim_m12.peak_params_[0, 2]
bw_sim_l = sim_l.peak_params_[0, 2]

freq_sim_s = sim_s.peak_params_[0, 0]
freq_sim_m = sim_m.peak_params_[0, 0]
freq_sim_m12 = sim_m12.peak_params_[0, 0]
freq_sim_l = sim_l.peak_params_[0, 0]

# %% B Calc IRASA
N_h = 16
N_h = 5  # increase in the end


# h_max3 = 40
irasa_params1 = dict(sf=srate_sim, band=band, win_sec=4,
                     hset=np.linspace(1.1, h_max1, N_h))
irasa_params2 = dict(sf=srate_sim, band=band, win_sec=4,
                     hset=np.linspace(1.1, h_max2, N_h))
irasa_params3 = dict(sf=srate_sim, band=band, win_sec=4,
                     hset=np.linspace(1.1, h_max3, N_h))


IRASA_sim_small_h1 = irasa(peak_small, **irasa_params1)

IRASA_sim_med_h1 = irasa(peak_med, **irasa_params1)
IRASA_sim_med_h2 = irasa(peak_med, **irasa_params2)
IRASA_sim_med12_h1 = irasa(peak_med, **irasa_params1)
IRASA_sim_med12_h2 = irasa(peak_med, **irasa_params2)

IRASA_sim_large_h1 = irasa(peak_large, **irasa_params1)
IRASA_sim_large_h2 = irasa(peak_large, **irasa_params2)
IRASA_sim_large_h3 = irasa(peak_large, **irasa_params3)

freqs_sim_s, ap_sim_small_h1, per1_s, params1_s = IRASA_sim_small_h1

freqs_sim_m, ap_sim_med_h1, per1_m, params1_m = IRASA_sim_med_h1
freqs_sim_m, ap_sim_med_h2, per2_m, params2_m = IRASA_sim_med_h2
freqs_sim_m12, ap_sim_med12_h1, per1_m12, params1_m12 = IRASA_sim_med12_h1
freqs_sim_m12, ap_sim_med12_h2, per2_m12, params2_m12 = IRASA_sim_med12_h2

freqs_sim_l, ap_sim_large_h1, per1_l, params1_l = IRASA_sim_large_h1
freqs_sim_l, ap_sim_large_h2, per2_l, params2_l = IRASA_sim_large_h2
freqs_sim_l, ap_sim_large_h3, per3_l, params3_l = IRASA_sim_large_h3

# normalize
ap_sim_small_h1 = ap_sim_small_h1[0]
ap_sim_med_h1 = ap_sim_med_h1[0]
ap_sim_med_h2 = ap_sim_med_h2[0]
ap_sim_med12_h1 = ap_sim_med12_h1[0]
ap_sim_med12_h2 = ap_sim_med12_h2[0]
ap_sim_large_h1 = ap_sim_large_h1[0]
ap_sim_large_h2 = ap_sim_large_h2[0]
ap_sim_large_h3 = ap_sim_large_h3[0]

plot_ap_sim_small_h1 = (freqs_sim_s, ap_sim_small_h1, c_IRASA1)

plot_ap_sim_med_h1 = (freqs_sim_m, ap_sim_med_h1, c_IRASA1)
plot_ap_sim_med_h2 = (freqs_sim_m, ap_sim_med_h2/10, c_IRASA2)
plot_ap_sim_med12_h1 = (freqs_sim_m12, ap_sim_med12_h1, c_IRASA1)
plot_ap_sim_med12_h2 = (freqs_sim_m12, ap_sim_med12_h2/10, c_IRASA2)

plot_ap_sim_large_h1 = (freqs_sim_l, ap_sim_large_h1, c_IRASA1)
plot_ap_sim_large_h2 = (freqs_sim_l, ap_sim_large_h2/10, c_IRASA2)
plot_ap_sim_large_h3 = (freqs_sim_l, ap_sim_large_h3/100, c_IRASA3)


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

# d
ylabel_d = r"$\left(T/m\right)^2$/Hz"

# e
ylabel_e = r"$T^2/Hz$"

# f
ylabel_f = r"$\mu V^2/Hz$"

# b
xlim_b = (1, 600)


def annotate_fit_range(ax, xmin, xmax, height, ylow=None, yhigh=None,
                       annotate_middle=True, annotate_range=True):
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
    if annotate_middle:
        box_alpha = 1
        ha = "center"
        text_pos = 10**((np.log10(xmin) + np.log10(xmax)) / 2)
    else:
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
    if annotate_range:
        ax.text(text_pos, height, s=f"{xmin:.1f}-{xmax:.1f}Hz", **anno_dic)
    else:
        ax.text(text_pos, height, s=f"{xmax-xmin:.1f}Hz", **anno_dic)
    if ylow and yhigh:
        ax.vlines(xmin, height, ylow, **vline_dic)
        ax.vlines(xmax, height, yhigh, **vline_dic)


# %% Plot
fig = plt.figure(figsize=[width, 6.9], constrained_layout=True)

gs0 = gridspec.GridSpec(3, 1, figure=fig)

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

gs02 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[2])
ax8 = fig.add_subplot(gs02[0])
ax9 = fig.add_subplot(gs02[1])
ax10 = fig.add_subplot(gs02[2])
# =============================================================================
# 
# # a
# # a1
# ax = axA1
# 
# # Plot sim
# ax.loglog(freq_a, toy_psd_a, c_sim)
# ax.loglog(freq0, psd_aperiodic_a, c_ap, zorder=0)
# 
# # Annotate fitting ranges
# vline_dic = dict(ls="--", clip_on=False, alpha=.3)
# ymin = ylim_a1[0]
# for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
#     y = toy_psd_a[freq_low]
#     xmin = freq_low
#     xmax = upper_fitting_border
#     coords = (y, xmin, xmax)
#     ax.hlines(*coords, color=color, ls="--")
#     v_coords = (xmin, ymin, y)
#     ax.vlines(*v_coords, color=color, **vline_dic)
# 
#     # Add annotation
#     s = f"{freq_low}-{xmax}Hz"
#     if i == 0:
#         s = "Fitting range: " + s
#         y = y**.97
#     else:
#         y = y**.98
#     ax.text(s=s, y=y, **text_dic)
# 
# # Set axes
# ax.text(s="a", **abc, transform=ax.transAxes)
# y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
# ax.yaxis.set_minor_locator(y_minor)
# ax.set_yticklabels([], minor=True)
# ax.set(**axes_a1)
# ax.set_ylabel(ylabel_a1, labelpad=-8)
# 
# # a2
# ax = axA2
# 
# # Plot error
# ax.semilogx(*error_plot_a)
# 
# # Annotate fitting ranges
# for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
#     xmin = freq_low
#     ymin = 0
#     ymax = 1.2
#     v_coords = (xmin, ymin, ymax)
#     ax.vlines(*v_coords, color=color, **vline_dic)
# 
# # Set axes
# ax.set(**axes_a2)
# 
# # b
# # b1
# ax = axB1
# 
# # Plot sim
# ax.loglog(freq_a, toy_psd_b, c_sim)
# ax.loglog(freq0, psd_aperiodic_b, c_ap, zorder=0)
# 
# # Annotate fitting ranges
# vline_dic = dict(ls="--", clip_on=False, alpha=.3)
# ymin = ylim_a1[0]
# for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
#     y = toy_psd_b[freq_low]
#     xmin = freq_low
#     xmax = upper_fitting_border
#     coords = (y, xmin, xmax)
#     ax.hlines(*coords, color=color, ls="--")
#     v_coords = (xmin, ymin, y)
#     ax.vlines(*v_coords, color=color, **vline_dic)
# 
#     # Add annotation
#     s = f"{freq_low}-{xmax}Hz"
#     if i == 0:
#         s = "Fitting range: " + s
#         y = y**.97
#     else:
#         y = y**.98
#     ax.text(s=s, y=y, **text_dic)
# 
# # Set axes
# # ax.text(s="b", **abc, transform=ax.transAxes)
# y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
# ax.yaxis.set_minor_locator(y_minor)
# ax.set_yticklabels([], minor=True)
# ax.set(**axes_b1)
# 
# # b2
# ax = axB2
# 
# # Plot error
# ax.semilogx(*error_plot_b)
# 
# # Annotate fitting ranges
# for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
#     xmin = freq_low
#     ymin = 0
#     ymax = 1.2
#     v_coords = (xmin, ymin, ymax)
#     ax.vlines(*v_coords, color=color, **vline_dic)
# 
# # Set axes
# ax.set(**axes_b2)
# 
# 
# # c
# # c1
# ax = axC1
# 
# # Plot sim
# ax.loglog(freq_a, toy_psd_c, c_sim)
# ax.loglog(freq0, psd_aperiodic_c, c_ap, zorder=0)
# 
# # Annotate fitting ranges
# vline_dic = dict(ls="--", clip_on=False, alpha=.3)
# ymin = ylim_a1[0]
# for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
#     y = toy_psd_c[freq_low]
#     xmin = freq_low
#     xmax = upper_fitting_border
#     coords = (y, xmin, xmax)
#     ax.hlines(*coords, color=color, ls="--")
#     v_coords = (xmin, ymin, y)
#     ax.vlines(*v_coords, color=color, **vline_dic)
# 
#     # Add annotation
#     s = f"{freq_low}-{xmax}Hz"
#     if i == 0:
#         s = "Fitting range: " + s
#         y = y**.97
#     else:
#         y = y**.98
#     ax.text(s=s, y=y, **text_dic)
# 
# # Set axes
# # ax.text(s="c", **abc, transform=ax.transAxes)
# y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
# ax.yaxis.set_minor_locator(y_minor)
# ax.set_yticklabels([], minor=True)
# ax.set(**axes_b1)
# 
# # c2
# ax = axC2
# 
# # Plot error
# ax.semilogx(*error_plot_c)
# 
# # Annotate fitting ranges
# for i, (freq_low, color) in enumerate(zip(freqs123, colors123)):
#     xmin = freq_low
#     ymin = 0
#     ymax = 1.2
#     v_coords = (xmin, ymin, ymax)
#     ax.vlines(*v_coords, color=color, **vline_dic)
# 
# # Set axes
# ax.set(**axes_b2)
# =============================================================================



# d
ax = ax5
ax.loglog(*plot_psd_small)
ax.loglog(*plot_ap_sim_small_h1, label=r"$h_{max}$ = "f"{h_max1}")

# annotate freq bandwidth
xmin = freq_sim_s - bw_sim_s
xmax = freq_sim_s + bw_sim_s
ylow = plot_psd_small[1][np.argmin(np.abs(plot_psd_small[0] - xmin))]
yhigh = plot_psd_small[1][np.argmin(np.abs(plot_psd_small[0] - xmax))]
height = 1e-14
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=False, annotate_range=False)

ax.legend(loc=1)
ax.set_ylabel(ylabel_a1)
ax.text(s="b", **abc, transform=ax.transAxes)
ax.set_xlim(xlim_b)
# ax.set_ylim((1e-6, 1))

# e
ax = ax6
ax.loglog(*plot_psd_med)
# ax.loglog(*plot_ap_sim_med_h1, label=r"$h_{max}$ = "f"{h_max1}")
# ax.loglog(*plot_ap_sim_med_h2, label=r"$h_{max}$ = "f"{h_max2}")
ax.loglog(*plot_ap_sim_med12_h1, label=r"$h_{max}$ = "f"{h_max1}")
ax.loglog(*plot_ap_sim_med12_h2, label=r"$h_{max}$ = "f"{h_max2}")
# ax.loglog(*plot_ap3_m, label=r"$h_{max}$ = "f"{h_max3}")
ax.set_xlim(xlim_b)
# ax.set_ylim((1e-6, 1))
height = 1e-14
xmin = freq_sim_m12 - bw_sim_m12
xmax = freq_sim_m12 + bw_sim_m12
ylow = plot_psd_small[1][np.argmin(np.abs(plot_psd_small[0] - xmin))]
yhigh = plot_psd_small[1][np.argmin(np.abs(plot_psd_small[0] - xmax))]
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=True, annotate_range=False)

ax.legend(loc=1)

# f
ax = ax7
ax.loglog(*plot_psd_large)
ax.loglog(*plot_ap_sim_large_h1, label=r"$h_{max}$ = "f"{h_max1}")
ax.loglog(*plot_ap_sim_large_h2, label=r"$h_{max}$ = "f"{h_max2}")
ax.loglog(*plot_ap_sim_large_h3, label=r"$h_{max}$ = "f"{h_max3}")

ax.set_xlim(xlim_b)
# ax.set_ylim((1e-6, 1))

# annotate freq bandwidth
xmin = freq_sim_l - bw_sim_l * 3
xmax = freq_sim_l + bw_sim_l * 3
height = 1e-15
ylow = plot_psd_small[1][np.argmin(np.abs(plot_psd_small[0] - xmin))]
yhigh = plot_psd_small[1][np.argmin(np.abs(plot_psd_small[0] - xmax))]
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=False, annotate_range=False)

ax.legend(loc=1)









# g
ax = ax8
ax.loglog(*plot_small)
ax.loglog(*plot_ap_small_h1, label=r"$h_{max}$ = "f"{h_max1}")
# ax.loglog(*plot_ap_small49, label="h_max=4.9")
ax.set_ylabel(ylabel_d)
ax.set_xlim(xlim_b)
ax.text(s="c", **abc, transform=ax.transAxes)
# annotate freq bandwidth
xmin = freq_real_s - bw_real_s
xmax = freq_real_s + bw_real_s
ylow = plot_small[1][np.argmin(np.abs(plot_small[0] - xmin))]
yhigh = plot_small[1][np.argmin(np.abs(plot_small[0] - xmax))]
height = 5e-26
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=False, annotate_range=False)



# h
ax = ax9
ax.loglog(*plot_med)
ax.loglog(*plot_ap_med19, label="h_max=1.9")
# ax.loglog(*plot_ap_med49, label="h_max=4.9")
ax.set_ylabel(ylabel_e)
ax.set_xlim(xlim_b)
# annotate freq bandwidth
xmin = freq_real_m1 - bw_real_m1
xmax = freq_real_m2 + bw_real_m2
ylow = plot_med[1][np.argmin(np.abs(plot_med[0] - xmin))]
yhigh = plot_med[1][np.argmin(np.abs(plot_med[0] - xmax))]
height = 1e-29
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=True, annotate_range=False)



# i
ax = ax10
ax.loglog(*plot_large)
ax.loglog(*plot_ap_large19, label="h_max=1.9")
# ax.loglog(*plot_ap_large49, label="h_max=4.9")
ax.set_ylabel(ylabel_f)
ax.set_xlim(xlim_b)
# annotate freq bandwidth
xmin = freq_real_l - bw_real_l
xmax = freq_real_l + bw_real_l
ylow = plot_large[1][np.argmin(np.abs(plot_large[0] - xmin))]
yhigh = plot_large[1][np.argmin(np.abs(plot_large[0] - xmax))]
height = 2e-3
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=False, annotate_range=False)



plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()

"""
Should I plot the oscillation extractions vs oscillations ground truths?

Real data: Show aperiodic component for two choices of h_max: In one case,
h_max is chosen to avoid highpass and noise floor
in the other h_max is chosen to identify peaks.

Maybe add fits

Replace medium peak of empirical data with sharper alpha peak. Too big. Dirty
business with fooof double peak and seldom seen in other data.
"""