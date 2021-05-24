"""
The evaluated frequency range is larger than the fitting range.

Rule: Avoid noise floor + highpass range!
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
from fooof.sim.gen import gen_aperiodic
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


# %% PARAMETERS

# Signal params
srate = 2400
nperseg = 1*srate  # 4*srate too high resolution for fooof
welch_params = dict(fs=srate, nperseg=nperseg)

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig4_IRASA_FreqRange.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# Litvak file names
path = "../data/Fig4/"
fname13 = "subj13_on_R4_raw.fif"

# Colors

# a)
c_sim = "k"
c_error = "r"
c_noise = "darkgray"

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

# %% a Sim Toy Signal with Three Oscillations and Fit

# fit in all frequency ranges from 1 to 80...
lower_fitting_borders = range(1, 80)
# ... to 100 Hz
upper_fitting_border = 100

# Oscillations parameters:
freq1, freq2, freq3 = 5, 15, 25  # Hz
amp1, amp2, amp3 = .7, .13, .02
width1, width2, width3 = .01, .01, .01
toy_slope = 2

periodic_params = [(freq1, amp1, width1),
                   (freq2, amp2, width2),
                   (freq3, amp3, width3)]

# Sim Toy Signal
_, toy_signal = osc_signals(toy_slope, periodic_params=periodic_params)

# welch_params["nperseg"] = srate

freq_a, toy_psd_a = sig.welch(toy_signal, **welch_params)

# Filter 1-100Hz
filt_a = (freq_a <= 100)
freq_a = freq_a[filt_a]
toy_psd_a = toy_psd_a[filt_a]

# %% a IRASA (takes very long)

# Fit IRASA and subtract ground truth to obtain fitting error
fit_errors = []
i = 0
for low in lower_fitting_borders:
    freq_range = (low, upper_fitting_border)
    _, _, _, params = irasa(data=toy_signal, band=freq_range, sf=srate)
    exp = -params["Slope"][0]
    error = np.abs(toy_slope - exp)
    fit_errors.append(error)
    print(f"{i/len(lower_fitting_borders):.2f}%")
    i += 1

error_plot_a = (lower_fitting_borders, fit_errors, c_error)

# %% B

# Make noise
slope_b = 2
noise_params_b = dict(slope=slope_b, nlv=0, highpass=True, seed=3)
pink2, _ = osc_signals(**noise_params_b)

# Calc PSD
freq_b, psd2_noise_b = sig.welch(pink2, **welch_params)

# Mask above highpass and below lowpass
filt_b = (freq_b < 140)
freq_b, psd2_noise_b = freq_b[filt_b], psd2_noise_b[filt_b]

# %% B Calc IRASA

# plot IRASA fits for increasing h

# plot fooof as ground truth
freq_range = (2, 30)
irasa_params = dict(sf=srate, band=freq_range, win_sec=4)

freq_range = irasa_params["band"]
fm = FOOOF(verbose=False)
fm.fit(freq_b, psd2_noise_b, freq_range)
fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
label = f"a(fooof)={fm.aperiodic_params_[1]:.2f}"
fooof_kwargs = dict(label=label)
plot_fooof_b = (fm.freqs, 10**fit, "b--")

IR_fit_plot_args_b = []
IR_fit_plot_kwargs_b = []

IR_plot_eff_args_b = []
IR_plot_eff_kwargs_b = []

h_maxima = [2, 8, 14, 20]
h_maxima = [2, 11, 20]
for h_max in h_maxima:
    # no oscillations, no harmonics, h can be integer
    N_h = 5  # not more needed for such simple sim
    IRASA = irasa(data=pink2, hset=np.linspace(1.1, h_max, N_h),
                  **irasa_params)
    freq_IR, _, _, IR_fit = IRASA
    IR_slope = -IR_fit["Slope"][0]
    IR_offset = IR_fit["Intercept"][0]

    IR_fit = gen_aperiodic(freq_IR, (IR_offset, IR_slope))
    label = fr"a(h$\leq${h_max})={IR_slope:.2f}"
    IR_kwargs = dict(label=label)
    plot_IRASA = (freq_IR, 10**IR_fit, "r--")
    IR_fit_plot_args_b.append(plot_IRASA)
    IR_fit_plot_kwargs_b.append(IR_kwargs)

    f_min = irasa_params["band"][0] / h_max
    f_max = irasa_params["band"][1] * h_max
    f_step = 1 / irasa_params["win_sec"]
    freq_IR_eff = np.arange(f_min, f_max, f_step)
    IR_fit_eff = gen_aperiodic(freq_IR_eff, (IR_offset, IR_slope))
    plot_IRASA_eff = (freq_IR_eff, 10**IR_fit_eff, "r--")

    IR_eff_kwargs = dict(alpha=0.2)
    IR_plot_eff_args_b.append(plot_IRASA_eff)
    IR_plot_eff_kwargs_b.append(IR_eff_kwargs)

# %% C

# Make noise
noise_params_c = dict(slope=slope_b, nlv=0.00003, highpass=False, seed=3)
pink2, _ = osc_signals(**noise_params_c)

# Calc PSD
welch_params["nperseg"] = 4*srate  # show lowpass filter
freq_c, psd2_noise_c = sig.welch(pink2, **welch_params)

# Detect Noise floor
floor_c = detect_noise_floor(freq_c, psd2_noise_c, f_start=1)
signal_c = (freq_c <= floor_c)
noise_c = (freq_c >= floor_c)

# plot fooof as ground truth
fm = FOOOF(verbose=False)
fm.fit(freq_c, psd2_noise_c, freq_range)
fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
label = f"a(fooof)={fm.aperiodic_params_[1]:.2f}"
fooof_kwargs = dict(label=label)
plot_fooof_c = (fm.freqs, 10**fit, "b--")

# %% C Calc IRASA

IR_fit_plot_args_c = []
IR_fit_plot_kwargs_c = []

IR_ap_plot_args_c = []
IR_ap_plot_kwargs_c = []

IR_plot_eff_args_c = []
IR_plot_eff_kwargs_c = []

for h_max in h_maxima:
    # Calc IRASA
    # no oscillations, no harmonics, h can be integer
    N_h = 5  # not more needed for such simple sim
    IRASA = irasa(data=pink2, hset=np.linspace(1.1, h_max, N_h),
                  **irasa_params)

    # Extract results
    freq_IR, _, IR_ap, IR_fit = IRASA
    IR_slope = -IR_fit["Slope"][0]
    IR_offset = IR_fit["Intercept"][0]

    # Make fit
    IR_fit = gen_aperiodic(freq_IR, (IR_offset, IR_slope))

    # Pack for plotting
    label = fr"a(h$\leq${h_max})={IR_slope:.2f}"
    IR_kwargs = dict(label=label)
    plot_IRASA = (freq_IR, 10**IR_fit, "r--")
    IR_fit_plot_args_c.append(plot_IRASA)
    IR_fit_plot_kwargs_c.append(IR_kwargs)

    # Calc effective freq range and pack for plotting
    f_min = irasa_params["band"][0] / h_max
    f_max = irasa_params["band"][1] * h_max
    f_step = 1 / irasa_params["win_sec"]
    freq_IR_eff = np.arange(f_min, f_max, f_step)
    IR_fit_eff = gen_aperiodic(freq_IR_eff, (IR_offset, IR_slope))
    plot_IRASA_eff = (freq_IR_eff, 10**IR_fit_eff, "r--")
    IR_eff_kwargs = dict(alpha=0.2)
    IR_plot_eff_args_c.append(plot_IRASA_eff)
    IR_plot_eff_kwargs_c.append(IR_eff_kwargs)

    # Pack aperiodic component for plotting
    plot_IRASA_ap = (freq_IR, IR_ap[0], "b--")
    # IRASA_ap_kwargs = dict(ls="--", color="b")
    IR_ap_plot_args_c.append(plot_IRASA_ap)
    # IR_ap_plot_kwargs.append(IRASA_ap_kwargs)





# %% D Real Spectrum

# Load Subj 13 MEG Right M1 On
# sub13 = mne.io.read_raw_fif(path + fname13, preload=True)
path = "/Users/moritzgerster/Documents/Code/Litvak11/data/1_raw_clean_fif/rest/"
sub = 6
cond = "off"
#fname13 = f"subj13/off/subj13_off_R1_raw.fif"
fname13 = f"subj{sub}/{cond}/subj{sub}_{cond}_R1_raw.fif"
sub13 = mne.io.read_raw_fif(path + fname13, preload=True)

# Select channel
# sub13.pick_channels(['leftM1'])
sub13.pick_channels(['SMA'])

# Filter out line noise
freq_filter = np.r_[np.arange(50, 601, 50), np.arange(60, 601, 60)]
filter_params = {"freqs": freq_filter,
                 "notch_widths": None,
                 "method": "spectrum_fit"}

sub13.notch_filter(**filter_params)

# Convert mne to numpy
srate = 2400
start = int(0.5 * srate)  # artefact in beginning of recording
stop = int(185 * srate)  # artefact at the end of recording

sub13 = sub13.get_data(start=start, stop=stop)[0]

# Calc Welch
welch_params = dict(fs=srate, nperseg=4*srate)  # detrend=False?

freq, psd_sub13 = sig.welch(sub13, **welch_params)

# Mask above highpass and below lowpass
filt = (freq <= 600)
freq = freq[filt]
psd_sub13 = psd_sub13[filt]

# %% D Calc IRASA

"""Load different PSD where fooof != IRASA for low freq range!"""

band_low = (1, 30)
band_high = (30, 45)

h_max = 2.9  # default
win_sec = 4  # default
f_min_low = band_low[0] / h_max
f_max_low = band_low[1] * h_max
f_min_high = band_high[0] / h_max
f_max_high = band_high[1] * h_max
f_step = 1 / win_sec

band_low_eff = (band_low[0] * h_max, band_low[1] / h_max)

hset = np.linspace(1.1, h_max, N_h)

IRASA_low = irasa(data=sub13, sf=srate, band=band_low, hset=hset)
IRASA_high = irasa(data=sub13, sf=srate, band=band_high, hset=hset)

IRASA_low_eff = irasa(data=sub13, sf=srate, band=band_low_eff, hset=hset)

freq_I_low, psd_aperiodic_low, psd_oscillatory_low, fit_params_low = IRASA_low
freq_I_high, psd_aperiodic_high, psd_oscillatory_high, fit_params_high = IRASA_high

freq_I_low_eff, psd_aperiodic_low_eff, psd_oscillatory_low_eff, fit_params_low_eff = IRASA_low_eff

psd_aperiodic_low = psd_aperiodic_low[0]
psd_oscillatory_low = psd_oscillatory_low[0]
psd_aperiodic_high = psd_aperiodic_high[0]
psd_oscillatory_high = psd_oscillatory_high[0]

psd_aperiodic_low_eff = psd_aperiodic_low_eff[0]
psd_oscillatory_low_eff = psd_oscillatory_low_eff[0]

IR_slope_low = -fit_params_low["Slope"][0]
IR_slope_high = -fit_params_high["Slope"][0]
IR_slope_low_eff = -fit_params_low_eff["Slope"][0]

label_low = fr"a low={IR_slope_low:.2f}"
label_high = fr"a high={IR_slope_high:.2f}"
label_low_eff = fr"a low-eff={IR_slope_low_eff:.2f}"

IR_offset_low = fit_params_low["Intercept"][0]
IR_offset_high = fit_params_high["Intercept"][0]
IR_offset_low_eff = fit_params_low_eff["Intercept"][0]

IR_fit_low = gen_aperiodic(freq_I_low, (IR_offset_low, IR_slope_low))
IR_fit_high = gen_aperiodic(freq_I_high, (IR_offset_high, IR_slope_high))
IR_fit_low_eff = gen_aperiodic(freq_I_low_eff, (IR_offset_low_eff,
                                                IR_slope_low_eff))

plot_IRASA_low = (freq_I_low, 10**IR_fit_low, "r--")
plot_IRASA_high = (freq_I_high, 10**IR_fit_high, "r--")
plot_IRASA_low_eff = (freq_I_low_eff, 10**IR_fit_low_eff, "r--")

freq_IR_eff_low = np.arange(f_min_low, f_max_low, f_step)
freq_IR_eff_high = np.arange(f_min_high, f_max_high, f_step)
freq_IR_eff_low_eff = np.arange(band_low[0], band_low[1], f_step)

IR_fit_eff_low = gen_aperiodic(freq_IR_eff_low, (IR_offset_low, IR_slope_low))
IR_fit_eff_high = gen_aperiodic(freq_IR_eff_high, (IR_offset_high, IR_slope_high))
IR_fit_eff_low_eff = gen_aperiodic(freq_IR_eff_low_eff, (IR_offset_low_eff,
                                                         IR_slope_low_eff))

plot_IRASA_eff_low = (freq_IR_eff_low, 10**IR_fit_eff_low, "r--")
plot_IRASA_eff_high = (freq_IR_eff_high, 10**IR_fit_eff_high, "r--")
plot_IRASA_eff_low_eff = (freq_IR_eff_low_eff, 10**IR_fit_eff_low_eff, "r--")

fm.fit(freq, psd_sub13, band_low)
fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
label = f"a low(fooof)={fm.aperiodic_params_[1]:.2f}"
fooof_kwargs_low = dict(label=label)
plot_fooof_low = (fm.freqs, 10**fit, "b--")

fm.fit(freq, psd_sub13, band_high)
fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
label = f"a high(fooof)={fm.aperiodic_params_[1]:.2f}"
fooof_kwargs_high = dict(label=label)
plot_fooof_high = (fm.freqs, 10**fit, "b--")


# %% Plot Params

width = 7.25  # inches
panel_fontsize = 12
legend_fontsize = 9
label_fontsize = 9
tick_fontsize = 9
annotation_fontsize = 8

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
xticklabels_a1 = []
yticks_a = [1e-10, 1e-7]
yticklabels_a1 = []
xlim_a = (1, 100)
ylabel_a1 = "PSD [a.u.]"
labelpad = 5
axes_a1 = dict(xticklabels=xticklabels_a1, yticks=yticks_a, xlim=xlim_a)
freqs123 = [freq1, freq2, freq3]
colors123 = [c_range1, c_range2, c_range3]
text_dic = dict(x=100, ha="right", fontsize=annotation_fontsize)

# a2
xticks_a2 = [1, 10, 100]
yticks_a2 = [0, 1]
xlabel_a2 = "Lower fitting range border [Hz]"
ylabel_a2 = "Fitting error"
ylim_a2 = (0, 1)
axes_a2 = dict(xticks=xticks_a2, xticklabels=xticks_a2, yticks=yticks_a2,
               xlim=xlim_a, xlabel=xlabel_a2, ylim=ylim_a2)

# b
xticks_b = [.1, 1, 10, 100, 600]
xlabel_b = "Frequency [Hz]"
yticks_b = [1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
yticklabels_b = [r"$10^{-13}$", "", "", "", "", "", r"$10^{-7}$"]
axes_b = dict(xticks=xticks_b, xlabel=xlabel_b, xticklabels=xticks_b,
              yticks=yticks_b, yticklabels=yticklabels_b,
              ylabel=None)

# c
axes_c = dict(xticks=xticks_b, xlabel=xlabel_b, xticklabels=xticks_b,
              yticks=yticks_b, yticklabels=[],
              ylabel=None)

tiny_leg = dict(labelspacing=.1, borderaxespad=0, handletextpad=.1,
                handlelength=1.5, borderpad=.1)

# d
ylim_d = (1, 5000)
yticks_d = [1, 10, 100, 1000]
yticklabels_d = [r"$10^1$", "", "", r"$10^3$"]
axes_d = dict(xticks=xticks_b, xlabel=xlabel_b, xticklabels=xticks_b,
              ylabel=None, ylim=ylim_d, yticks=yticks_d,
              yticklabels=yticklabels_d)
ticks_psd = dict(length=4, width=1)

# e
ylim_e = (3, 6000)
xlim_e = (0.2, 600)
axes_e = dict(xticks=xticks_b, xlabel=xlabel_b, xticklabels=xticks_b,
              ylabel=None, ylim=ylim_e, xlim=xlim_e)

# f
ylim_f = (2, 200)
xlim_f = (8, 200)
axes_f = dict(xticks=xticks_b, xlabel=xlabel_b, xticklabels=xticks_b,
              ylabel=None, ylim=ylim_f, xlim=xlim_f)
# %%
fig = plt.figure(figsize=[width, 6], constrained_layout=True)

gs0 = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[3, 3, 7])

# =============================================================================
# ax1 = fig.add_subplot(gs0[0, 0])
# ax2 = fig.add_subplot(gs0[1, 0])
# ax3 = fig.add_subplot(gs0[:2, 1])
# ax4 = fig.add_subplot(gs0[:2, 2])
#
# ax5 = fig.add_subplot(gs0[2, 0])
# ax6 = fig.add_subplot(gs0[2, 1])
# ax7 = fig.add_subplot(gs0[2, 2])
# =============================================================================

gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4, 5])

gs00 = gs0[0].subgridspec(2, 3)
ax1 = fig.add_subplot(gs00[0, 0])
ax2 = fig.add_subplot(gs00[1, 0])
ax3 = fig.add_subplot(gs00[:, 1])
ax4 = fig.add_subplot(gs00[:, 2])

gs01 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[1])
ax5 = fig.add_subplot(gs01[0])
ax6 = fig.add_subplot(gs01[1])
ax7 = fig.add_subplot(gs01[2])

# a
# a1
ax = ax1

# Plot sim
ax.loglog(freq_a, toy_psd_a, c_sim)

# Annotate fitting ranges
vline_dic = dict(ls="--", clip_on=False, alpha=0.3)
ymin, ymax = ax.get_ylim()
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
ax.set(**axes_a1, ylim=(ymin, ymax))
ax.set_ylabel(ylabel_a1, labelpad=labelpad)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels(yticklabels_a1)

# a2
ax = ax2

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
ax.set_ylabel(ylabel_a2, labelpad=0)

# b
ax = ax3
ax.loglog(freq_b, psd2_noise_b, c_sim)
ax.loglog(*plot_fooof_b, **fooof_kwargs)
for plot_IRASA, IR_kwargs in zip(IR_fit_plot_args_b, IR_fit_plot_kwargs_b):
    ax.loglog(*plot_IRASA, **IR_kwargs)

for plot_IRASA_eff, IR_kwargs_eff in zip(IR_plot_eff_args_b,
                                         IR_plot_eff_kwargs_b):
    ax.loglog(*plot_IRASA_eff, **IR_kwargs_eff)

ax.legend(**tiny_leg)
ax.text(s="b", **abc, transform=ax.transAxes)
ax.set(**axes_b)
ax.set_ylabel(ylabel_a1, labelpad=-15)
ax.tick_params(**ticks_psd)
x_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.xaxis.set_minor_locator(x_minor)
x_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(x_minor)

# c
ax = ax4
ax.loglog(freq_c[signal_c], psd2_noise_c[signal_c], c_sim)
ax.loglog(freq_c[noise_c], psd2_noise_c[noise_c], c_noise)
ax.loglog(*plot_fooof_c, **fooof_kwargs)
for plot_IRASA, IR_kwargs in zip(IR_fit_plot_args_c, IR_fit_plot_kwargs_c):
    ax.loglog(*plot_IRASA, **IR_kwargs)

for plot_IRASA_eff, IR_kwargs_eff in zip(IR_plot_eff_args_c,
                                         IR_plot_eff_kwargs_c):
    ax.loglog(*plot_IRASA_eff, **IR_kwargs_eff)
ax.legend(**tiny_leg)
ax.set(**axes_c)
ax.text(s="c", **abc, transform=ax.transAxes)
ax.tick_params(**ticks_psd)
x_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.xaxis.set_minor_locator(x_minor)

# d
ax = ax5
ax.loglog(freq, psd_sub13, c_real, label=f"MEG Sub. {sub}")
ax.loglog(*plot_fooof_low, **fooof_kwargs_low, lw=4)
ax.legend()
ax.tick_params(**ticks_psd)
ax.text(s="d", **abc, transform=ax.transAxes)
ax.set(**axes_d)
ax.set_ylabel(r"PSD [$\mu$$V^2$/Hz]", labelpad=-10)


# e
ax = ax6
ax.loglog(freq, psd_sub13, c_real)
# ax.loglog(freq_I_low, psd_aperiodic_low, label="Aperiodic Component")

ax.loglog(*plot_IRASA_low_eff, label=label_low_eff, lw=4)
ax.loglog(*plot_IRASA_eff_low_eff, alpha=.3, lw=4)
ax.loglog(*plot_IRASA_low, label=label_low, lw=4)
ax.loglog(*plot_IRASA_eff_low, alpha=.3, lw=4)

ax.legend()
ax.tick_params(**ticks_psd)
ax.text(s="e", **abc, transform=ax.transAxes)
ax.set(**axes_e)

# f

ax = ax7
ax.loglog(freq, psd_sub13, c_real)

ax.loglog(*plot_fooof_high, **fooof_kwargs_high, lw=4)
ax.loglog(*plot_IRASA_high, label=label_high, lw=4)
ax.loglog(*plot_IRASA_eff_high, alpha=.3, lw=4)

ax.legend()
ax.tick_params(**ticks_psd)
ax.text(s="f", **abc, transform=ax.transAxes)
ax.set(**axes_f)

plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()
"""
To do:
    - eliminate yticklabels a1
    - correct 600Hz xticklabel c
    - make nice d-f
    - set yticklabels e and f
"""