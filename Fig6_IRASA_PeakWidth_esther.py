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
from helper import irasa
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


# =============================================================================
# def calc_error(signal):
#     """Fit IRASA and subtract ground truth to obtain fitting error."""
#     fit_errors = []
#     for i in trange(len(lower_fitting_borders)):
#         freq_range = (lower_fitting_borders[i], upper_fitting_border)
#         _, _, _, params = irasa(data=signal, band=freq_range, sf=srate)
#         exp = -params["Slope"][0]
#         error = np.abs(toy_slope - exp)
#         fit_errors.append(error)
#     return fit_errors
# 
# =============================================================================

# %% PARAMETERS

# Signal params
srate = 2400
nperseg = srate
welch_params = dict(fs=srate, nperseg=nperseg)

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig6_PeakWidth_Esther.pdf"
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
c_IRASA3 = "C5"

lw = 2


# %% C Real data

srate = 2400
nperseg = srate
welch_params = dict(fs=srate, nperseg=nperseg)

highpass = .3  # Hz
lowpass = 600  # Hz

fname_MEG = "Subj016_ON_1_raw.fif"
fname_LFP = "Subj016_OFF_001_STN_r+s_raw.fif"

sub_MEG = mne.io.read_raw_fif(path + fname_MEG, preload=True)
sub_LFP = mne.io.read_raw_fif(path + fname_LFP, preload=True)

# ch2 = "MEG2621" # beta peak overlay: 0433
small_grad = "MEG0913"
med_mag = "MEG0113"
large_LFP = "R2-R3"

channels = [small_grad, med_mag]

sub_MEG.pick_channels(channels, ordered=True)
sub_LFP.pick_channels([large_LFP])

# Convert to numpy
MEG_raw = sub_MEG.get_data()
LFP_raw = sub_LFP.get_data()[0]
LFP_raw *= 1e6  # convert V to uV


freq, spec_MEG = sig.welch(MEG_raw, **welch_params)
freq, spec_LFP = sig.welch(LFP_raw, **welch_params)


filt = (freq <= 1200)
freq_filt = freq[filt]
spec_GRAD = spec_MEG[0, filt]
spec_MAG = spec_MEG[1, filt]


plot_small = (freq_filt, spec_GRAD, c_real)
plot_med = (freq_filt, spec_MAG, c_real)
plot_large = (freq, spec_LFP, c_real)

# plot_small_low = (freq, spec_MEG[0]/100, c_real)
plot_med_low = (freq_filt, spec_MAG/10, c_real)
plot_large_low = (freq, spec_LFP/10, c_real)

# plot_small_lower = (freq, spec_MEG[0]/100, c_real)
# plot_med_lower = (freq, spec_MEG[1]/100, c_real)
# plot_large_lower = (freq, spec_LFP/10000, c_real)

# %% C Apply fooof to determine peak width

real_s = FOOOF(max_n_peaks=2, verbose=False)
real_m = FOOOF(max_n_peaks=1, verbose=False, peak_width_limits=(0.5, 150))
# real_m12 = FOOOF(max_n_peaks=2, verbose=False, peak_width_limits=(0.5, 150))
real_l = FOOOF(max_n_peaks=1, verbose=False, peak_width_limits=(.5, 150))

real_s.fit(freq, spec_MEG[0], (3, 20))
real_m.fit(freq, spec_MEG[1], (1, 100))
# real_m12.fit(freq, spec_MEG[1], (1, 100))
real_l.fit(freq, spec_LFP, (10, 150))

# real_m.plot(plt_log=True)

freq_real_s, pow_real_s, bw_real_s = real_s.peak_params_[0]
freq_real_m, pow_real_m, bw_real_m = real_m.peak_params_[0]
# freq_real_m1, pow_real_m1, bw_real_m1 = real_m12.peak_params_[0]
# freq_real_m2, pow_real_m2, bw_real_m2 = real_m12.peak_params_[1]
freq_real_l, pow_real_l, bw_real_l = real_l.peak_params_[0]

# %% C Calc IRASA
h_max1 = 2
h_max2 = 25
# h_max3 = 45

# doesn't matter that shorter band makes more sence, this is topic of fig4
band = (1, 100)

band_h1 = (highpass * h_max1, lowpass / h_max1)
band_h2 = (highpass * h_max2, lowpass / h_max2)
# band_h3 = (highpass * h_max3, lowpass / h_max3)

N_h = 16
# N_h = 5  # increase in the end

irasa_params1 = dict(sf=srate, band=band_h1, hset=np.linspace(1.1, h_max1, N_h))
irasa_params2 = dict(sf=srate, band=band_h2, hset=np.linspace(1.1, h_max2, N_h))
# irasa_params3 = dict(sf=srate, band=band_h3, hset=np.linspace(1.1, h_max3, N_h))

IRASA_small_h1 = irasa(MEG_raw[0], **irasa_params1)
IRASA_med_h1 = irasa(MEG_raw[1], **irasa_params1)
IRASA_large_h1 = irasa(LFP_raw, **irasa_params1)
IRASA_m_h2 = irasa(MEG_raw[1], **irasa_params2)
IRASA_l_h2 = irasa(LFP_raw, **irasa_params2)
# IRASA_l_h3 = irasa(LFP_raw, **irasa_params3)


freq_I_h1, ap_small_h1, per_small_h1, params_small_h1 = IRASA_small_h1
_, ap_med_h1, per_med_h1, params_med_h1 = IRASA_med_h1
_, ap_large_h1, per_large_h1, params_large_h1 = IRASA_large_h1
freq_I_h2, ap_med_h2, per_med_h2, params_med_h2 = IRASA_m_h2
freq_I_h2, ap_large_h2, per_large_h2, params_large_h2 = IRASA_l_h2
# freq_I_h3, ap_large_h3, per_large_h3, params_large_h3 = IRASA_l_h3


plot_ap_small_h1 = (freq_I_h1, ap_small_h1[0], c_IRASA1)
plot_ap_med_h1 = (freq_I_h1, ap_med_h1[0], c_IRASA1)
plot_ap_large_h1 = (freq_I_h1, ap_large_h1[0], c_IRASA1)
plot_ap_med_h2 = (freq_I_h2, ap_med_h2[0]/10, c_IRASA2)
plot_ap_large_h2 = (freq_I_h2, ap_large_h2[0]/10, c_IRASA2)
# plot_ap_large_h3 = (freq_I_h3, ap_large_h3[0]/10000, c_IRASA3)


# Show what happens for larger freq ranges
irasa_params2["band"] = band_h1
# irasa_params3["band"] = band_h1

IRASA_med_h1_long = irasa(MEG_raw[1], **irasa_params1)
IRASA_large_h1_long = irasa(LFP_raw, **irasa_params1)
IRASA_m_h2_long = irasa(MEG_raw[1], **irasa_params2)
IRASA_l_h2_long = irasa(LFP_raw, **irasa_params2)
# IRASA_l_h3_long = irasa(LFP_raw, **irasa_params3)

freq_I_long, ap_med_h1_long, per_med_h1_long, params_med_h1_long = IRASA_med_h1_long
_, ap_large_h1_long, per_large_h1_long, params_large_h1_long = IRASA_large_h1_long
_, ap_med_h2_long, per_med_h2_long, params_med_h2_long = IRASA_m_h2_long
_, ap_large_h2_long, per_large_h2_long, params_large_h2_long = IRASA_l_h2_long
# _, ap_large_h3_long, per_large_h3_long, params_large_h3_long = IRASA_l_h3_long

plot_ap_med_h1_long = (freq_I_long, ap_med_h1_long[0], c_IRASA1)
plot_ap_large_h1_long = (freq_I_long, ap_large_h1_long[0], c_IRASA1)
plot_ap_med_h2_long = (freq_I_long, ap_med_h2_long[0]/10, c_IRASA2)
plot_ap_large_h2_long = (freq_I_long, ap_large_h2_long[0]/10, c_IRASA2)
# plot_ap_large_h3_long = (freq_I_long, ap_large_h3_long[0]/10000, c_IRASA3)


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

# d
ylabel_d = r"$\left(T/m\right)^2$/Hz"

# e
ylabel_e = r"$T^2/Hz$"

# f
ylabel_f = r"$\mu V^2/Hz$"

# b
xlim_b = (None, 700)
xlabel = "Frequency [Hz]"
xticks_b = [1, 10, 100, 1000]

# c
xticks_c = [1, 10, 100, 600]


def annotate_range(ax, xmin, xmax, height, ylow=None, yhigh=None,
                   annotate_pos=True, annotation="log-diff",
                   annotation_fontsize=7):
    """
    Annotate fitting range or peak width.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Ax to draw the lines.
    xmin : float
        X-range minimum.
    xmax : float
        X-range maximum.
    height : float
        Position on y-axis of range.
    ylow : float, optional
        Position on y-axis to connect the vertical lines. If None, no vertical
        lines are drawn. The default is None.
    yhigh : float, optional
        Position on y-axis to connect the vertical lines. If None, no vertical
        lines are drawn. The default is None.
    annotate_pos : bool, optional
        Where to annotate.
    annotate : bool, optional
        The kind of annotation.
        "diff": Print range.
        "log-diff": Print range and logrange.
        else: Print range1-range2

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
        xdiff = xmax - xmin
        if xdiff > 50:  # round large intervals
            xdiff = np.round(xdiff, -1)
        range_str = (r"$\Delta f=$"f"{xdiff:.0f}Hz\n"
                     r"$\Delta f_{log}=$"f"{(np.log10(xmax/xmin)):.1f}")
    else:
        range_str = f"{xmin:.1f}-{xmax:.1f}Hz"
    ax.text(text_pos, text_height, s=range_str, **anno_dic)

    if ylow and yhigh:
        ax.vlines(xmin, height, ylow, **vline_dic)
        ax.vlines(xmax, height, yhigh, **vline_dic)

# %% Plot
fig, axes = plt.subplots(1, 3, figsize=[width, 2.7])


alpha_long = .3


# c1
ax = axes[0]
ax.loglog(*plot_small, label="Grad")
ax.loglog(*plot_ap_small_h1, label=r"$h_{max}$ = "f"{h_max1}")
# ax.loglog(*plot_ap_fit_small_h1, label=f"a={small_h1_slope:.2f}")
ax.set_ylabel(ylabel_d)
ax.set_xlabel(xlabel)
# ax.set_xlim(xlim_b)
ymin, ymax = ax.get_ylim()
ax.set_ylim((3e-26, ymax))
ax.text(s="a", **abc, transform=ax.transAxes)
# annotate freq bandwidth
xmin = freq_real_s - bw_real_s
xmax = freq_real_s + bw_real_s
ylow = plot_small[1][np.argmin(np.abs(plot_small[0] - xmin))]
yhigh = plot_small[1][np.argmin(np.abs(plot_small[0] - xmax))]
height = 6e-26
annotate_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
               height=height, annotate_pos="left")
ax.set_xticks(xticks_c)
ax.set_xticklabels(xticks_c)
ax.legend(loc=1, borderaxespad=0)


# c2
ax = axes[1]
ax.loglog(*plot_med, label="Mag")
ax.loglog(*plot_ap_med_h1)#, label=r"$h_{max}$ = "f"{h_max1}")
ax.loglog(*plot_ap_med_h1_long, ls="--", alpha=alpha_long)
# ax.loglog(*plot_ap_fit_med_h1, label=f"a={med_h1_slope:.2f}")

ax.loglog(*plot_med_low, alpha=.5)
ax.loglog(*plot_ap_med_h2, label=r"$h_{max}$ = "f"{h_max2}")
ax.loglog(*plot_ap_med_h2_long, ls="--", alpha=alpha_long)
# ax.loglog(*plot_ap_fit_med_h2, label=f"a={med_h2_slope:.2f}")
ax.set_ylabel(ylabel_e)
ax.set_xlabel(xlabel)
ax.text(s="b", **abc, transform=ax.transAxes)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
# ax.set_xlim(xlim_b)
ylim = (3e-27, 5e-23)
ax.set_ylim(ylim)
ax.legend(loc=1, borderaxespad=0)
# annotate freq bandwidth
xmin = freq_real_m - bw_real_m + 2
xmax = freq_real_m + bw_real_m + 2
ylow = plot_med_low[1][np.argmin(np.abs(plot_med_low[0] - xmin))]
yhigh = plot_med_low[1][np.argmin(np.abs(plot_med_low[0] - xmax))]
height = 5 * ylim[0]
annotate_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
               height=height, annotate_pos=.4)
ax.set_xticks(xticks_b)
ax.set_xticklabels(xticks_b)

# c3
ax = axes[2]
ax.loglog(*plot_large, label="LFP")
ax.loglog(*plot_ap_large_h1)#, label=r"$h_{max}$ = "f"{h_max1}")
ax.loglog(*plot_ap_large_h1_long, ls="--", alpha=alpha_long)
# ax.loglog(*plot_ap_fit_large_h1, label=f"a={large_h1_slope:.2f}")

ax.loglog(*plot_large_low, alpha=.5)
ax.loglog(*plot_ap_large_h2)#, label=r"$h_{max}$ = "f"{h_max2}")
ax.loglog(*plot_ap_large_h2_long, ls="--", alpha=alpha_long)
# ax.loglog(*plot_ap_fit_large_h2, label=f"a={large_h2_slope:.2f}")

#ax.loglog(*plot_large_lower, alpha=.5)
#ax.loglog(*plot_ap_large_h3, label=r"$h_{max}$ = "f"{h_max3}")
#ax.loglog(*plot_ap_large_h3_long, ls="--", alpha=alpha_long)
# ax.loglog(*plot_ap_fit_large_h3, label=f"a={large_h3_slope:.2f}")
ax.set_ylabel(ylabel_f)
ax.set_xlabel(xlabel)
# ax.set_xlim(xlim_b)
ax.legend(loc=1, borderaxespad=0)
ylim = (1e-5, 2)
ax.set_ylim(ylim)
ax.text(s="c", **abc, transform=ax.transAxes)
x_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.xaxis.set_minor_locator(x_minor)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)

# annotate freq bandwidth
xmin = freq_real_l - bw_real_l
xmax = freq_real_l + bw_real_l
ylow = plot_large_low[1][np.argmin(np.abs(plot_large_low[0] - xmin))]
yhigh = plot_large_low[1][np.argmin(np.abs(plot_large_low[0] - xmax))]
height = 10 * ylim[0]
annotate_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
               height=height, annotate_pos="left")
ax.set_xticks(xticks_b)
ax.set_xticklabels(xticks_b)
# ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(fig_path + fig_name[:-4] + "SuppMat.pdf", bbox_inches="tight")
plt.show()
