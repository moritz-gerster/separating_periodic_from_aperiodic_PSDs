"""Explain fooof and IRASA."""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig
from pathlib import Path
from fooof.sim.gen import gen_aperiodic
from fooof.plts.spectra import plot_spectrum
# from fooof.plts.annotate import plot_annotated_peak_search
from fooof_fit_MG import FOOOF
from fooof_annotate_MG import plot_annotated_peak_search_MG
# from fooof_fm_MG import plot_fm_lin_MG
# import matplotlib.gridspec as gridspec
from noise_helper import irasa
import fractions
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


# %% PARAMETERS

# Signal params
srate = 2400
win_sec = 2
welch_params_f = dict(fs=srate, nperseg=2*srate)
welch_params_I = dict(fs=srate, nperseg=win_sec*srate)

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig0_Intro.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)


# Colors

# a)
c_sim = "k"
c_ap = "tab:brown"
c_osc = "b"
c_full = "r"
c_fit = "white"

# Boxes
c_inp = "#73DCFF"
c_alg = "#FFC6FF"
c_mod = "#FFFF7E" # tab:orange, darkorange
c_sub = "#9EFFA2"
ls_box = (0, (5, 7))
ls_box = "-"
lw_box = .5
alpha_box = .6

c_inp_dark = "deepskyblue"
c_inp_dark = "k" # new
c_alg_dark = "#FF66FF"
c_mod_dark = "y"
c_sub_dark = "g"

# Algorithm
c_h1 = c_inp
c_h2 = "yellow"
c_h3 = "g"

c_tresh = c_h2
c_flat = "orange"


# temporary colors
c_inp = "w"
c_alg = "w"
c_mod = "w"
c_sub = "w"
c_sim = "k"
c_ap = "tab:brown"
c_osc = "b"
c_full = "r"
c_fit = "grey"
c_inp_dark = "k"
c_alg_dark = "k"
c_mod_dark = "k"
c_sub_dark = "k"
#c_h1 = "k"
#c_h2 = "k"
#c_h3 = "k"
#c_tresh = "k"
#c_flat = "k"

lw = 2
lw_fit = 3
ls_fit = (0, (3, 3))
lw_h = 1
#ls_fit = "-."


# %% Make signal
toy_slope = 1
freq1 = 10  # Hz
freq2 = 25  # Hz
amp1 = .02
amp2 = .01
width = .001

periodic_params = [(freq1, amp1, width)]#,
                   # (freq2, amp2, width)]

# Sim Toy Signal
toy_aperiodic, toy_comb = osc_signals(toy_slope,
                                      periodic_params=periodic_params,
                                      highpass=False)

toy_aperiodic, toy_comb = toy_aperiodic*1e6, toy_comb*1e6

#_, toy_osc = osc_signals(0,
 #                        periodic_params=periodic_params,
  #                       highpass=False)

freqs_f, psd_comb_f = sig.welch(toy_comb, **welch_params_f)
#_, psd_osc_f = sig.welch(toy_osc, **welch_params_f)
freqs_I, psd_comb_I = sig.welch(toy_comb, **welch_params_I)
#_, psd_osc_I = sig.welch(toy_osc, **welch_params_I)

# Filter 1-100Hz
mask_f = (freqs_f <= 100) & (freqs_f >= 1)
freqs_f = freqs_f[mask_f]
psd_comb_f = psd_comb_f[mask_f]
#psd_osc_f = psd_osc_f[mask_f]

mask_I = (freqs_I <= 100) & (freqs_I >= 1)
freqs_I = freqs_I[mask_I]
psd_comb_I = psd_comb_I[mask_I]
#psd_osc_I = psd_osc_I[mask_I]

# %% Create dummy signal for illustration
time = np.arange(0, len(toy_comb)/srate, 1/srate)

time_series = np.sin(2 * np.pi * 10 * time)
# Add 1/f
time_series += .01 * toy_aperiodic

# %% Calc fooof

# Set whether to plot in log-log space
plt_log = True
freq_range = (1, 100)

fm = FOOOF(max_n_peaks=1) # use default settings
fm.add_data(freqs_f, psd_comb_f, freq_range)

# Fit the power spectrum model
fm.fit(freqs_f, psd_comb_f, freq_range)

# Do an initial aperiodic fit - a robust fit, that excludes outliers
init_ap_fit = gen_aperiodic(fm.freqs,
                            fm._robust_ap_fit(fm.freqs, fm.power_spectrum))

# Recompute the flattened spectrum using the initial aperiodic fit
init_flat_spec = fm.power_spectrum - init_ap_fit
init_flat_spec_lin = 10**fm.power_spectrum - 10**init_ap_fit

# %% Calc IRASA
# hset = np.arange(1.1, 1.95, 0.05)
hset = [1.1, 1.5, 2]

irasa_params = dict(sf=srate, band=freq_range,
                    win_sec=win_sec, hset=hset)

IRASA = irasa(data=toy_comb, **irasa_params)

freqs_irasa, psd_ap, psd_osc, params = IRASA

psd_ap, psd_osc = psd_ap[0], psd_osc[0]


IR_offset = params["Intercept"][0]
IR_slope = -params["Slope"][0]

psd_fit = gen_aperiodic(freqs_irasa, (IR_offset, IR_slope))

hset = [1.1, 1.5, 2]

win = welch_params_I["nperseg"]
psds_resampled = np.zeros((len(hset), *psd_comb_I.shape))

for i, h in enumerate(hset):
    # Get the upsampling/downsampling (h, 1/h) factors as integer
    rat = fractions.Fraction(str(h))
    up, down = rat.numerator, rat.denominator
    # Much faster than FFT-based resampling
    data_up = sig.resample_poly(toy_comb, up, down)
    data_down = sig.resample_poly(toy_comb, down, up, axis=-1)

    freqs_up, psd_up = sig.welch(data_up, h * srate, nperseg=win)
    freqs_dw, psd_dw = sig.welch(data_down, srate / h, nperseg=win)
    
    psds_resampled[i, :] = np.sqrt(psd_up * psd_dw)[mask_I]

# Now we take the median PSD of all the resampling factors, which gives
# a good estimate of the aperiodic component of the PSD.
psd_median = np.median(psds_resampled, axis=0)


# %% Plot Params


fig_width = 7.25  # inches
panel_fontsize = 12
legend_fontsize = 7
# label_fontsize = 1
# tick_fontsize = 1
# annotation_fontsize = tick_fontsize

# mpl.rcParams['xtick.labelsize'] = tick_fontsize
# mpl.rcParams['ytick.labelsize'] = tick_fontsize
# mpl.rcParams['axes.labelsize'] = label_fontsize
# mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['font.size'] = panel_fontsize
mpl.rcParams['legend.fontsize'] = legend_fontsize
mpl.rcParams["axes.spines.right"] = True
mpl.rcParams["axes.spines.top"] = True
mpl.rcParams["axes.spines.left"] = True
mpl.rcParams["axes.spines.bottom"] = True


yticks = [10, 100, 1000]
yticks_small = [1, 5]
yticks_lin = [0, 1000, 2000]

# yticks_lin_f = [0, 300, 600]
# ylim_lin_f = [-100, 600]
yticks_lin_f = [0, .5]
ylim_lin_f = [-.1, 1]

yticks_lin_I = [0, 1000, 2000]
ylim_lin_I = [-100, yticks_lin_I[-1]]


def input_series(ax, duration=1, step=srate//100):
    mask = [(time >= duration) & (time <= 2*duration)]
    ax.plot(time[mask][::step], time_series[mask][::step])
    ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
    ax.axis("off")
#    ax.text(x=-.17, y=.5, s="Input", va="center", ha="right",
#            c=c_inp_dark, transform=ax.transAxes)
    # ax.set_title("Input", c=c_inp_dark, y=1)
# =============================================================================
#     ax.spines["right"].set_linestyle(ls_box)
#     ax.spines["left"].set_linestyle(ls_box)
#     ax.spines["top"].set_linestyle(ls_box)
#     ax.spines["bottom"].set_linestyle(ls_box)
#     ax.spines["right"].set_linewidth(lw_box)
#     ax.spines["left"].set_linewidth(lw_box)
#     ax.spines["top"].set_linewidth(lw_box)
#     ax.spines["bottom"].set_linewidth(lw_box)
#     ax.set_facecolor(c_inp)
#     ax.patch.set_alpha(alpha_box)
# =============================================================================
    # ax.legend(handlelength=0, borderpad=.2)


def input_psd(ax):
    ax.loglog(freqs_f, psd_comb_f, c_sim)
    ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
    ax.set_yticks([], minor=True)
    ax.set_xticks([], minor=True)
    ax.set_yticklabels([], minor=True)
    ax.set_xticklabels([], minor=True)
    ax.axis("off")
# =============================================================================
#     ax.spines["right"].set_linestyle(ls_box)
#     ax.spines["left"].set_linestyle(ls_box)
#     ax.spines["top"].set_linestyle(ls_box)
#     ax.spines["bottom"].set_linestyle(ls_box)
#     ax.spines["right"].set_linewidth(lw_box)
#     ax.spines["left"].set_linewidth(lw_box)
#     ax.spines["top"].set_linewidth(lw_box)
#     ax.spines["bottom"].set_linewidth(lw_box)
#     ax.set_facecolor(c_inp)
#     ax.patch.set_alpha(alpha_box)
# =============================================================================
    # ax.legend(handlelength=0, borderpad=.2)
    fm.plot


# =============================================================================
# def fooof_1(ax):    
#     f_res = welch_params_f["fs"] / welch_params_f["nperseg"]
#     ax.loglog(freqs_f, psd_comb_f, c_sim)
#     ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
#     ax.set_yticks([], minor=True)
#     ax.set_xticks([], minor=True)
#     ax.set_yticklabels([], minor=True)
#     ax.set_xticklabels([], minor=True)
#     ax.set_title("FOOOF")
#     ax.spines["right"].set_linestyle(ls_box)
#     ax.spines["left"].set_linestyle(ls_box)
#     ax.spines["top"].set_linestyle(ls_box)
#     ax.spines["bottom"].set_linestyle(ls_box)
#     ax.spines["right"].set_linewidth(lw_box)
#     ax.spines["left"].set_linewidth(lw_box)
#     ax.spines["top"].set_linewidth(lw_box)
#     ax.spines["bottom"].set_linewidth(lw_box)
#     ax.set_facecolor(c_inp)
#     ax.patch.set_alpha(alpha_box)
#     # ax.legend(handlelength=0, borderpad=.2)
#     fm.plot
# #    ax.rcParams["axes.spines.top"] = True
#  #   ax.rcParams["axes.spines.left"] = True
#   #  ax.rcParams["axes.spines.bottom"] = True
# =============================================================================


def fooof_1(ax):
    plot_spectrum(fm.freqs, 10**fm.power_spectrum, log_freqs=False,
                  # label='Original Power Spectrum',
                  color=c_sim, ax=ax)
    plot_spectrum(fm.freqs, 10**init_ap_fit, log_freqs=False,
                  label='Initial Fit',
                  color=c_fit, lw=lw_fit, alpha=1, ls=ls_fit, ax=ax)
    ax.grid(False)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis("off")
    # ax.set_title("FOOOF", y=.9)
    leg = ax.legend(handlelength=2, handletextpad=.5, loc=0)
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 1, 0))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2)
        legobj.set_linestyle((0, (2.5, 2)))
    # ax.legend(fontsize=legend_fontsize)
    

def fooof_2(ax):
    plot_spectrum(fm.freqs, init_flat_spec, log_freqs=False,
                  label='Flattened PSD', color=c_flat, ax=ax)
    ax.set_xscale("log")
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis("off")
    ylim = ax.get_ylim()
    # ax.set_ylim(ylim_lin_f)
    ax.get_legend().remove()
    leg = ax.legend()
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 1, 0))
    return ylim
    
    
def fooof_3(ax, ylim):
    plot_annotated_peak_search_MG(fm, 0, ax, lw=lw, markersize=10,
                                  c_flat=c_flat, c_gauss=c_osc,
                                  c_thresh=c_tresh)
    ax.set_xscale("log")
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis("off")
    ax.set_ylim(ylim)
    leg = ax.legend(loc=(.6, .7))
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 1, 0))
  #  for legobj in leg.legendHandles:
 ##       legobj.set_linewidth(2)
 #       legobj.set_linestyle((0, (2.5, 2)))
    ax.set_title(None)


def fooof_4(ax, ylim):
    plot_annotated_peak_search_MG(fm, 1, ax, lw=lw, markersize=10,
                                  c_flat=c_flat, c_gauss=c_osc, c_thresh=c_tresh)
    ax.set_xscale("log")
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis("off")
    ax.set_ylim(ylim)
    # ax.legend(fontsize=legend_fontsize)
    ax.set_title(None)


def aperiodic(ax):
    plot_spectrum(fm.freqs, 10**fm._spectrum_peak_rm, log_freqs=False,
                  label='Aperiodic PSD', color=c_ap, lw=lw_fit, ax=ax)
    plot_spectrum(fm.freqs, 10**fm._ap_fit, log_freqs=False,
                  label='Aperiodic Fit', lw=lw_fit,
                  color=c_fit, alpha=1, ls=ls_fit, ax=ax)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
    ax.set_yticks([], minor=True)
    ax.set_xticks([], minor=True)
    ax.set_yticklabels([], minor=True)
    ax.set_xticklabels([], minor=True)
    leg = ax.legend()
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 1, 0))
    ax.axis("off")
# =============================================================================
#     ax.spines["right"].set_linestyle(ls_box)
#     ax.spines["left"].set_linestyle(ls_box)
#     ax.spines["top"].set_linestyle(ls_box)
#     ax.spines["bottom"].set_linestyle(ls_box)
#     ax.spines["right"].set_linewidth(lw_box)
#     ax.spines["left"].set_linewidth(lw_box)
#     ax.spines["top"].set_linewidth(lw_box)
#     ax.spines["bottom"].set_linewidth(lw_box)
#     ax.set_facecolor(c_mod)
# =============================================================================
    # ax.patch.set_alpha(alpha_box)


# =============================================================================
# def fooof_7(ax):
#     fm.plot_lin_MG(plt_log=False, plot_aperiodic=False,
#                    ax=ax, model_kwargs=dict(color=c_full, alpha=1, lw=lw),
#                    data_kwargs=dict(alpha=0),
#                    label="Fit +\nOscillatory")
#     ax.grid(False)
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.axis("off")
#     leg = ax.legend()
#     leg.get_frame().set_alpha(None)
#     leg.get_frame().set_facecolor((0, 0, 1, 0))
# =============================================================================


def oscillatory(ax):
    plot_spectrum(fm.freqs, fm._peak_fit, log_freqs=False, color=c_osc,
                  label='Oscillatory PSD', ax=ax)
    ax.grid(False)
    # ax.set_ylim([-.05, .75])
    ax.set_xscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis("off")
    ax.get_legend().remove()
    leg = ax.legend()
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 1, 0))


# =============================================================================
# def IRASA_1(ax):
#     f_res = welch_params_I["fs"] / welch_params_I["nperseg"]
#     ax.loglog(freqs_I, psd_comb_I, c_sim)
#     ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
#     ax.set_yticks([], minor=True)
#     ax.set_xticks([], minor=True)
#     ax.set_yticklabels([], minor=True)
#     ax.set_xticklabels([], minor=True)
#     # ax.legend(handlelength=0, borderpad=.2)
#     ax.set_title("IRASA")
#     ax.set_facecolor(c_inp)
#     ax.patch.set_alpha(alpha_box)
#     ax.spines["right"].set_linestyle(ls_box)
#     ax.spines["left"].set_linestyle(ls_box)
#     ax.spines["top"].set_linestyle(ls_box)
#     ax.spines["bottom"].set_linestyle(ls_box)
#     ax.spines["right"].set_linewidth(lw_box)
#     ax.spines["left"].set_linewidth(lw_box)
#     ax.spines["top"].set_linewidth(lw_box)
#     ax.spines["bottom"].set_linewidth(lw_box)
# =============================================================================

# =============================================================================
# def IRASA_2(ax):
#     ax.loglog(freqs_I, psd_comb_I, c_sim)
#     i = 0
#     ax.loglog(freqs_I, psds_resampled[i], c_h1, lw=lw_h, label=f"h={hset[i]}")
#     ax.axis("off")
#     ax.legend(handlelength=1)
#     # ax.set_xlabel("Frequency [Hz]")
#     ax.set_title("IRASA", y=.9)
# 
# 
# def IRASA_3(ax):        
#     ax.loglog(freqs_I, psd_comb_I, c_sim)
#     i = 0
#     ax.loglog(freqs_I, psds_resampled[i], c_h1, lw=lw_h, label=f"h={hset[i]}")
#     i = 1
#     ax.loglog(freqs_I, psds_resampled[i],  c_h2, lw=lw_h, label=f"h={hset[i]}")
#     ax.axis("off")
#     ax.legend(handlelength=1)
#     ax.set_xlabel("Frequency [Hz]")
# =============================================================================


# =============================================================================
# def IRASA_4(ax):
#     ax.loglog(freqs_I, psd_comb_I, c_sim)
#     i = 2
#     ax.loglog(freqs_I, psds_resampled[i],  c_h3, lw=lw_h, label=f"h={hset[i]}")
#     ax.axis("off")
#     ax.legend(handlelength=1)
# =============================================================================
    
    
def IRASA_5(ax):
    ax.loglog(freqs_I, psd_comb_I, c_sim, label="Original Data")
    for i, c in enumerate([c_h1, c_h2, c_h3]):
        ax.loglog(freqs_I, psds_resampled[i], c, lw=lw_h, label=f"h={hset[i]}")
    ymin, ymax = ax.get_ylim()
    freq = 5
    ax.annotate(f"{freq}Hz     ",
                xy=(freq, psd_comb_I[freqs_I==freq][0]),
                xytext=(freq, ymin*1.5), fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="-", lw=1, ls=":"))
    freq = 10
    ax.annotate(f"{freq}Hz",
                xy=(freq, psd_comb_I[freqs_I==freq][0]),
                xytext=(freq, ymin*1.5), fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="-", lw=1, ls=":"))
    freq = 20
    ax.annotate(f"       {freq}Hz",
                xy=(freq, psd_comb_I[freqs_I==freq][0]),
                xytext=(freq, ymin*1.5), fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="-", lw=1, ls=":"))
    ax.axis("off")
    # ax.set_title("IRASA", c=c_alg_dark, y=1)
    

# =============================================================================
# def IRASA_6(ax):
#     ax.loglog(freqs_I, psd_median, c_ap, lw=lw_fit, label="Aperiodic PSD")
#     ax.loglog(freqs_irasa, 10**psd_fit, c_fit, lw=lw_fit, ls=ls_fit, label="Aperiodic Fit")
#     ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
#     ax.set_yticks([], minor=True)
#     ax.set_xticks([], minor=True)
#     ax.set_yticklabels([], minor=True)
#     ax.set_xticklabels([], minor=True)
#     leg = ax.legend()
#     leg.get_frame().set_alpha(None)
#     leg.get_frame().set_facecolor((0, 0, 1, 0))
#     ax.spines["right"].set_linestyle(ls_box)
#     ax.spines["left"].set_linestyle(ls_box)
#     ax.spines["top"].set_linestyle(ls_box)
#     ax.spines["bottom"].set_linestyle(ls_box)
#     ax.spines["right"].set_linewidth(lw_box)
#     ax.spines["left"].set_linewidth(lw_box)
#     ax.spines["top"].set_linewidth(lw_box)
#     ax.spines["bottom"].set_linewidth(lw_box)
#     ax.set_facecolor(c_mod)
#     ax.patch.set_alpha(alpha_box)
# 
#     
#     
# def IRASA_7(ax):
#     ax.semilogx(freqs_irasa, psd_osc, c_osc, label="Oscillatory PSD")
#     ymin, ymax = ax.get_ylim()
#     ax.set_ylim(ymin, ymax*1.3)
#     ax.axis("off")
#     leg = ax.legend()#loc=(-.4, 1.05))
#     leg.get_frame().set_alpha(None)
#     leg.get_frame().set_facecolor((0, 0, 1, 0))
# 
# 
# def IRASA_8(ax):
#     # ax.loglog(freqs_I, psd_comb_I, c_sim, label="Original Data")
#     ax.loglog(freqs_irasa, psd_ap + psd_osc, c_full, label="Aperiodic PSD +\nOscillatory PSD")
#     ymin, ymax = ax.get_ylim()
#     ax.set_ylim(ymin, ymax*5)
#     ax.axis("off")
#     leg = ax.legend()
#     leg.get_frame().set_alpha(None)
#     leg.get_frame().set_facecolor((0, 0, 1, 0))
# =============================================================================


def make_frame(ax, c, **kwargs):    
    ax = fig.add_subplot(ax, **kwargs)
    ax.tick_params(axis='both', which='both',bottom=0, left=0,
                            labelbottom=0, labelleft=0)
    ax.set_facecolor(c)
    ax.patch.set_alpha(alpha_box)
    ax.spines["right"].set_linestyle(ls_box)
    ax.spines["left"].set_linestyle(ls_box)
    ax.spines["top"].set_linestyle(ls_box)
    ax.spines["bottom"].set_linestyle(ls_box)
    ax.spines["right"].set_linewidth(lw_box)
    ax.spines["left"].set_linewidth(lw_box)
    ax.spines["top"].set_linewidth(lw_box)
    ax.spines["bottom"].set_linewidth(lw_box)
    return ax





# %% Plot

"""
To do:
    - input: show time series, input to IRASA, PSD to fooof!
    - querformat    
    - add box names

    - add arrows and annotation steps
    - implement correct colors for all steps    
"""



fig = plt.figure(figsize=(fig_width, 4))


gs = fig.add_gridspec(nrows=2, ncols=3,
                      width_ratios=[2, 3, 2],
                      height_ratios=[2, 3],
                      hspace=.3, wspace=.3)


gs_input = gs[:, 0].subgridspec(5, 1, height_ratios=[1, 5, 1, 5, 1])
input_frame = make_frame(gs_input[1:4], c_alg)
input_frame.set_title("Input", c=c_alg_dark, y=1)

inp_ser = fig.add_subplot(gs_input[1], ymargin=1, xmargin=.1)
inp_PSD = fig.add_subplot(gs_input[3], ymargin=.1, xmargin=.1)


input_series(inp_ser, duration=.5, step=24)
input_psd(inp_PSD)

# Algorithm gs
irasa_frame = make_frame(gs[0, 1], c_alg)
irasa_frame.set_title("IRASA", c=c_alg_dark, y=1)

gs_IRASA = gs[0, 1].subgridspec(1, 1)
gs_IR = fig.add_subplot(gs_IRASA[0, 0], xmargin=.1, ymargin=.1)
IRASA_5(gs_IR)


gs_fooof = gs[1, 1].subgridspec(2, 2, hspace=.0, wspace=.0)
fooof_frame = make_frame(gs_fooof[:, :], c_alg)#, position=[0, 0, 1, 1],
#                         transform=gs_fooof[:, :].transAxes)
# gs_fooof.tight_layout(h_pad=1, v_pad=1)
fooof_frame.set_title("FOOOF", c=c_alg_dark, y=1)

margins = dict(xmargin=.3, ymargin=.3)
fooof1 = fig.add_subplot(gs_fooof[1, 0], **margins)
fooof2 = fig.add_subplot(gs_fooof[0, 0], **margins)
fooof3 = fig.add_subplot(gs_fooof[0, 1], **margins)
fooof4 = fig.add_subplot(gs_fooof[1, 1], **margins)

fooof_1(fooof1)
ylim = fooof_2(fooof2)
fooof_3(fooof3, ylim)
fooof_4(fooof4, ylim)

gs_output = gs[:, 2].subgridspec(4, 1, hspace=.3, height_ratios=[1, 3, 3, 1])

output_frame = make_frame(gs_output[1:3], c_alg)

# bbox = output_frame.figbox
# pos = output_frame.get_position
# output_frame.set_position([bbox.x0*.9, bbox.y0*.9, (bbox.x1-bbox.x0)*1.1, (bbox.y1 - bbox.y0)*1.1])

output_frame.set_title("Output", c=c_alg_dark, y=1)
ap = fig.add_subplot(gs_output[1], ymargin=.3, xmargin=.3)
osc = fig.add_subplot(gs_output[2], ymargin=.3, xmargin=.3)

aperiodic(ap)
oscillatory(osc)

# Add text
# fig.text(.1, 0.791, "Original data", fontsize=9, ha="left")
# fig.text(.1, 0.38, "Fit", ha="left", color=c_mod_dark)
# fig.text(.1, 0.275, "Model", ha="left", color=c_sub_dark) # "Input - Model"

plt.savefig(fig_path + "new.pdf", bbox_inches="tight")
plt.show()



# %% Plot Old

# =============================================================================
# 
# fig = plt.figure(constrained_layout=False, figsize=(fig_width, 9))
# 
# gs = fig.add_gridspec(nrows=4, ncols=4,
#                       height_ratios=[2, 3, 2, 2],
#                       width_ratios=[1, 1.5, 1.5, 1],
#                       wspace=.1, hspace=.3, left=.1, right=.9)
# 
# # 4 steps in infographics
# inp = fig.add_subplot(gs[0, 1:3])
# 
# # Algorithm gs
# make_frame(gs[1, :2], c_alg)
# gs_alg_f = gs[1, :2].subgridspec(2, 2, wspace=0, hspace=0)
# 
# gs_alg_f1 = fig.add_subplot(gs_alg_f[0, 1])
# gs_alg_f2 = fig.add_subplot(gs_alg_f[0, 0])
# gs_alg_f3 = fig.add_subplot(gs_alg_f[1, 0])
# gs_alg_f4 = fig.add_subplot(gs_alg_f[1, 1])
# 
# make_frame(gs[1, 2:], c_alg)
# gs_alg_I = gs[1, 2:].subgridspec(2, 2, wspace=0, hspace=0)
# 
# gs_alg_I1 = fig.add_subplot(gs_alg_I[0, 0])
# gs_alg_I2 = fig.add_subplot(gs_alg_I[0, 1])
# gs_alg_I3 = fig.add_subplot(gs_alg_I[1, 1])
# gs_alg_I4 = fig.add_subplot(gs_alg_I[1, 0])
# 
# # Model gs
# mod_f = fig.add_subplot(gs[2, 1])
# mod_I = fig.add_subplot(gs[2, 2])
# 
# # Subtraction gs
# make_frame(gs[3, :2], c_sub)
# gs_sub_f = gs[3, :2].subgridspec(1, 2, wspace=0, hspace=0)
# sub_f1 = fig.add_subplot(gs_sub_f[1])
# sub_f2 = fig.add_subplot(gs_sub_f[0])
# 
# make_frame(gs[3, 2:], c_sub)
# gs_sub_I = gs[3, 2:].subgridspec(1, 2, wspace=0, hspace=0)
# sub_I1 = fig.add_subplot(gs_sub_I[0])
# sub_I2 = fig.add_subplot(gs_sub_I[1])
# 
# 
# input_psd(inp)
# 
# 
# fooof_2(gs_alg_f1)
# fooof_3(gs_alg_f2)
# fooof_4(gs_alg_f3)
# fooof_5(gs_alg_f4)
# 
# IRASA_2(gs_alg_I1)
# IRASA_3(gs_alg_I2)
# IRASA_4(gs_alg_I3)
# IRASA_5(gs_alg_I4)
# 
# 
# fooof_6(mod_f)
# IRASA_6(mod_I)
# 
# 
# fooof_7(sub_f1)
# fooof_8(sub_f2)
# IRASA_7(sub_I1)
# IRASA_8(sub_I2)
# 
# # Add text
# fig.text(.1, 0.81, "Input", ha="left", color=c_inp_dark)
# # fig.text(.1, 0.791, "Original data", fontsize=9, ha="left")
# fig.text(.1, 0.706, "Peak removal", ha="left", color=c_alg_dark) # Algorithm
# fig.text(.1, 0.38, "Fit", ha="left", color=c_mod_dark)
# fig.text(.1, 0.275, "Model", ha="left", color=c_sub_dark) # "Input - Model"
# 
# 
# fig.text(.53, .46, "median", fontsize=10, ha="left")
# fig.text(.52, .27, "subtract\nfrom PSD", fontsize=10, ha="left")
# 
# # Add arrows
# """Find way to position arrows FIXED. They should not move after adding
# additional arrows."""
# # =============================================================================
# 
# x_large_f = .36
# x_large_I = .65
# 
# y_inp_tail = .735
# y_inp_head = .765
# 
# y_alg_tail = .43
# y_alg_head = .46
# 
# y_mod_tail = .21
# y_mod_head = .24
# 
# 
# arr_props = dict(facecolor='k', width=1.5, headwidth=6, headlength=5, shrink=0)
# 
# plt.annotate(text="", xy=(x_large_f, y_inp_tail),
#              xytext=(x_large_f, y_inp_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_large_I, y_inp_tail),
#              xytext=(x_large_I, y_inp_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# plt.annotate(text="", xy=(x_large_f, y_alg_tail),
#              xytext=(x_large_f, y_alg_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_large_I, y_alg_tail),
#              xytext=(x_large_I, y_alg_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# plt.annotate(text="", xy=(x_large_f, y_mod_tail),
#              xytext=(x_large_f, y_mod_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_large_I, y_mod_tail),
#              xytext=(x_large_I, y_mod_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# 
# plt.annotate(text="", xy=(.135, y_mod_tail),
#              xytext=(.135, y_alg_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# x_sub_tail_f = .23
# x_sub_head_f = .26
# 
# x_sub_tail_I = .74
# x_sub_head_I = .77
# 
# y_sub_small = .115
# y_alg_small1 = .66
# y_alg_small3 = .55
# 
# y_alg_head_I = .615
# y_alg_tail_I = .645
# 
# y_alg_head_f = .6
# y_alg_tail_f = .62
# 
# x_alg_f = .135
# x_alg_I = .865
# 
# plt.annotate(text="", xy=(x_sub_head_f, y_sub_small),
#              xytext=(x_sub_tail_f, y_sub_small),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_sub_head_I, y_sub_small),
#              xytext=(x_sub_tail_I, y_sub_small),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# plt.annotate(text="", xy=(x_sub_head_f, y_sub_small),
#              xytext=(x_sub_tail_f, y_sub_small),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_sub_head_I, y_sub_small),
#              xytext=(x_sub_tail_I, y_sub_small),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# plt.annotate(text="", xy=(x_sub_tail_f, y_alg_small1),
#              xytext=(x_sub_head_f, y_alg_small1),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_sub_head_I, y_alg_small1),
#              xytext=(x_sub_tail_I, y_alg_small1),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# plt.annotate(text="", xy=(x_alg_f, y_alg_head_f),
#              xytext=(x_alg_f, y_alg_tail_f),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_alg_I, y_alg_head_I),
#              xytext=(x_alg_I, y_alg_tail_I),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# plt.annotate(text="", xy=(x_sub_head_f, y_alg_small3),
#              xytext=(x_sub_tail_f, y_alg_small3),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_sub_tail_I, y_alg_small3),
#              xytext=(x_sub_head_I, y_alg_small3),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# plt.savefig(fig_path + "style_old.pdf", bbox_inches="tight")
# plt.show()
# =============================================================================

# =============================================================================
# 
# # %% Arrows
# 
# 
# fig = plt.figure(constrained_layout=False, figsize=(fig_width, 9))
# 
# gs = fig.add_gridspec(nrows=4, ncols=4,
#                       height_ratios=[2, 3, 2, 2],
#                       width_ratios=[1, 1.5, 1.5, 1],
#                       wspace=.1, hspace=.25, left=.1, right=.9)
# 
# # 4 steps in infographics
# inp_f = fig.add_subplot(gs[0, 1])
# inp_I = fig.add_subplot(gs[0, 2])
# 
# # Algorithm gs
# make_frame(gs[1, :2], c_alg)
# gs_alg_f = gs[1, :2].subgridspec(2, 2, wspace=0, hspace=0)
# 
# gs_alg_f1 = fig.add_subplot(gs_alg_f[0, 1])
# gs_alg_f2 = fig.add_subplot(gs_alg_f[0, 0])
# gs_alg_f3 = fig.add_subplot(gs_alg_f[1, 0])
# gs_alg_f4 = fig.add_subplot(gs_alg_f[1, 1])
# 
# make_frame(gs[1, 2:], c_alg)
# gs_alg_I = gs[1, 2:].subgridspec(2, 2, wspace=0, hspace=0)
# 
# gs_alg_I1 = fig.add_subplot(gs_alg_I[0, 0])
# gs_alg_I2 = fig.add_subplot(gs_alg_I[0, 1])
# gs_alg_I3 = fig.add_subplot(gs_alg_I[1, 1])
# gs_alg_I4 = fig.add_subplot(gs_alg_I[1, 0])
# 
# # Model gs
# mod_f = fig.add_subplot(gs[2, 1])
# mod_I = fig.add_subplot(gs[2, 2])
# 
# # Subtraction gs
# make_frame(gs[3, :2], c_sub)
# gs_sub_f = gs[3, :2].subgridspec(1, 2, wspace=0, hspace=0)
# sub_f1 = fig.add_subplot(gs_sub_f[1])
# sub_f2 = fig.add_subplot(gs_sub_f[0])
# 
# make_frame(gs[3, 2:], c_sub)
# gs_sub_I = gs[3, 2:].subgridspec(1, 2, wspace=0, hspace=0)
# sub_I1 = fig.add_subplot(gs_sub_I[0])
# sub_I2 = fig.add_subplot(gs_sub_I[1])
# 
# x_large_f = .36
# x_large_I = .65
# 
# y_inp_tail = .735
# y_inp_head = .765
# 
# y_alg_tail = .43
# y_alg_head = .46
# 
# y_mod_tail = .21
# y_mod_head = .24
# 
# 
# arr_props = dict(facecolor='k', width=1.5, headwidth=6, headlength=5, shrink=0)
# 
# plt.annotate(text="", xy=(x_large_f, y_inp_tail),
#              xytext=(x_large_f, y_inp_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_large_I, y_inp_tail),
#              xytext=(x_large_I, y_inp_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# plt.annotate(text="", xy=(x_large_f, y_alg_tail),
#              xytext=(x_large_f, y_alg_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_large_I, y_alg_tail),
#              xytext=(x_large_I, y_alg_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# plt.annotate(text="", xy=(x_large_f, y_mod_tail),
#              xytext=(x_large_f, y_mod_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_large_I, y_mod_tail),
#              xytext=(x_large_I, y_mod_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# 
# plt.annotate(text="", xy=(.135, y_mod_tail),
#              xytext=(.135, y_alg_head),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# 
# 
# 
# # =============================================================================
# # plt.annotate(text="", xy=(.135, .9),
# #              xytext=(.135, .1),
# #              xycoords='figure fraction',
# #              annotation_clip=False, arrowprops=arr_props)
# # =============================================================================
# 
# 
# 
# 
# x_sub_tail_f = .23
# x_sub_head_f = .26
# 
# x_sub_tail_I = .74
# x_sub_head_I = .77
# 
# y_sub_small = .115
# y_alg_small1 = .66
# y_alg_small3 = .55
# 
# y_alg_head_I = .615
# y_alg_tail_I = .645
# 
# y_alg_head_f = .6
# y_alg_tail_f = .62
# 
# x_alg_f = .135
# x_alg_I = .865
# 
# plt.annotate(text="", xy=(x_sub_head_f, y_sub_small),
#              xytext=(x_sub_tail_f, y_sub_small),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_sub_head_I, y_sub_small),
#              xytext=(x_sub_tail_I, y_sub_small),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# plt.annotate(text="", xy=(x_sub_head_f, y_sub_small),
#              xytext=(x_sub_tail_f, y_sub_small),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_sub_head_I, y_sub_small),
#              xytext=(x_sub_tail_I, y_sub_small),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# plt.annotate(text="", xy=(x_sub_tail_f, y_alg_small1),
#              xytext=(x_sub_head_f, y_alg_small1),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_sub_head_I, y_alg_small1),
#              xytext=(x_sub_tail_I, y_alg_small1),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# plt.annotate(text="", xy=(x_alg_f, y_alg_head_f),
#              xytext=(x_alg_f, y_alg_tail_f),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_alg_I, y_alg_head_I),
#              xytext=(x_alg_I, y_alg_tail_I),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# plt.annotate(text="", xy=(x_sub_head_f, y_alg_small3),
#              xytext=(x_sub_tail_f, y_alg_small3),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# plt.annotate(text="", xy=(x_sub_tail_I, y_alg_small3),
#              xytext=(x_sub_head_I, y_alg_small3),
#              xycoords='figure fraction',
#              annotation_clip=False, arrowprops=arr_props)
# 
# 
# plt.show()
# 
# =============================================================================
