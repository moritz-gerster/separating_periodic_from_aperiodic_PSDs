"""
The evaluated frequency range is larger than the fitting range.

Rule: Avoid noise floor + highpass range!

Figure:

    a: Show specturm similar to 2a. Move IRASA along frequency ranges and show
    small fitting error (maybe do the same for fooof and show with low alpha)
    Visualize effective frequency range as explanation

    b: Show noise floor error increases with increasing resampling.
    Fooof works fine (before noise floor)
    Plot Signal: as in 1a but noise floor 50Hz later
    Plot Fooof fit correct
    Plot IRASA fit error increases with h_set_max (maybe 4 lines)
    Plot IRASA aperiodic component for largest error.
    Consider: Maybe make 3 panels to avoid info overload

    c: Plot spectrum in low frequency range (maybe 1-20Hz)
    Plot fits in low range 1-30Hz with increasing h (no noise),
    show increasing fit error
    Show one aperiodic component for largest error
    Consider: 3 panels


PROBLEM: 3 PANELS AS ROW MAKE FIGURE TOO SMALL
3 PANELS AS COLUMN TOO LARGE

CONSIDER: split a in two panels or add another panel

ADD: add real spectrum where low pass filter causes trouble. Maybe MEG spectrum?
 add real spectrum where noise floor causes trouble. Maybe same MEG spectrum?


(Mention: Even without highpass problem because of diffferent 1/f bevhavor,
 see He et al. Neuron)

"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig
from pathlib import Path
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
nperseg = 4*srate  # 4*srate too high resolution for fooof
"""to do: check if they use hann window in original publication."""
welch_params = dict(fs=srate, nperseg=nperseg, window='hann')

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig4_IRASA_FreqRange.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

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

# %% a) Sim Toy Signal with Three Oscillations and Fit

# fit in all frequency ranges from 1 to 80...
lower_fitting_borders = range(1, 80)
# ... to 100 Hz
upper_fitting_border = 100

# Oscillations parameters:
freq1, freq2, freq3 = 5, 15, 25  # Hz
amp1, amp2, amp3 = 2.5, 1, .5
width1, width2, width3 = .01, .01, .01
toy_slope = 1
nlv = 0

periodic_params = [(freq1, amp1, width1),
                   (freq2, amp2, width2),
                   (freq3, amp3, width3)]

# Sim Toy Signal
_, toy_signal = osc_signals(toy_slope, periodic_params=periodic_params,
                            nlv=0, highpass=False)

# welch_params["nperseg"] = srate

freq_a, toy_psd_a = sig.welch(toy_signal, **welch_params)

# Filter 1-100Hz
filt_a = (freq_a > 0) & (freq_a <= 100)
freq_a = freq_a[filt_a]
toy_psd_a = toy_psd_a[filt_a]

toy_plot_a = (freq_a, toy_psd_a, c_sim)

# %% IRASA

# Fit IRASA and subtract ground truth to obtain fitting error
fit_errors = []
for low in lower_fitting_borders:
    freq_range = (low, upper_fitting_border)
    _, _, _, params = irasa(data=toy_signal, band=freq_range, sf=srate)
    exp = -params["Slope"][0]
    error = np.abs(toy_slope - exp)
    fit_errors.append(error)

error_plot_a = (lower_fitting_borders, fit_errors, c_error)


# %% B

# Make noise
slope_b = 2
noise_params_b = dict(slope=slope_b, nlv=0.00003, highpass=False, seed=3)
pink2, _ = osc_signals(**noise_params_b)

# Calc PSD
freq_b, psd2_noise_b = sig.welch(pink2, **welch_params)

# Mask above highpass and below lowpass
filt_b = (freq_b > 0) & (freq_b < 600)
freq_b, psd2_noise_b = freq_b[filt_b], psd2_noise_b[filt_b]

# Normalize
# psd2_noise /= psd2_noise[0]

# Detect Noise floor
floor_b = detect_noise_floor(freq_b, psd2_noise_b, f_start=1)
signal_b = (freq_b <= floor_b)
noise_b = (freq_b >= floor_b)


# plot fooof as ground truth
freq_range = (1, 30)
fm = FOOOF(verbose=False)
fm.fit(freq_b, psd2_noise_b, freq_range)
fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
label = r"$a_{fooof}$="f"{fm.aperiodic_params_[1]:.2f}"
fooof_kwargs = dict(label=label)
plot_fooof_b = (fm.freqs, 10**fit, "b--")

# %% Calc IRASA

# plot IRASA fits for increasing h
irasa_params = dict(sf=srate, band=(1, 30), win_sec=4)  # standard parameters

IR_fit_plot_args_b = []
IR_fit_plot_kwargs_b = []

IR_ap_plot_args_b = []
IR_ap_plot_kwargs_b = []

IR_plot_eff_args_b = []
IR_plot_eff_kwargs_b = []

for h_max in [2, 5, 10, 15]:
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
    # label = r"$a_{IRASA}$="f"{IR_slope:.2f}, "r"$h_{max}$="f"{h_max}"
    substr = f"hmax={h_max}"
    label = fr"$a_{{{substr}}}$={IR_slope:.2f}"
    IR_kwargs = dict(label=label)
    plot_IRASA = (freq_IR, 10**IR_fit, "r--")
    IR_fit_plot_args_b.append(plot_IRASA)
    IR_fit_plot_kwargs_b.append(IR_kwargs)

    # Calc effective freq range and pack for plotting
    f_min = irasa_params["band"][0] / h_max
    f_max = irasa_params["band"][1] * h_max
    f_step = 1 / irasa_params["win_sec"]
    freq_IR_eff = np.arange(f_min, f_max, f_step)
    IR_fit_eff = gen_aperiodic(freq_IR_eff, (IR_offset, IR_slope))
    plot_IRASA_eff = (freq_IR_eff, 10**IR_fit_eff, "r--")
    IR_eff_kwargs = dict(alpha=0.2)
    IR_plot_eff_args_b.append(plot_IRASA_eff)
    IR_plot_eff_kwargs_b.append(IR_eff_kwargs)

    # Pack aperiodic component for plotting
    plot_IRASA_ap = (freq_IR, IR_ap[0], "b--")
    # IRASA_ap_kwargs = dict(ls="--", color="b")
    IR_ap_plot_args_b.append(plot_IRASA_ap)
    # IR_ap_plot_kwargs.append(IRASA_ap_kwargs)


# %% C

# Make noise
slope_a = 2
noise_params_c = dict(slope=slope_a, nlv=0, highpass=True, seed=3)
pink2, _ = osc_signals(**noise_params_c)

# Calc PSD
welch_params["nperseg"] = 4*srate  # show lowpass filter
freq_c, psd2_noise_c = sig.welch(pink2, **welch_params)

# Mask above highpass and below lowpass
filt_c = (freq_c < 140)
freq_c, psd2_noise_c = freq_c[filt_c], psd2_noise_c[filt_c]

# %% Calc IRASA

# plot IRASA fits for increasing h
irasa_params = dict(sf=srate, band=(2, 30), win_sec=4)  # standard parameters

# plot fooof as ground truth
freq_range = irasa_params["band"]
fm = FOOOF(verbose=False)
fm.fit(freq_c, psd2_noise_c, freq_range)
fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
label = r"$a_{fooof}$="f"{fm.aperiodic_params_[1]:.2f}"
fooof_kwargs = dict(label=label)
plot_fooof_c = (fm.freqs, 10**fit, "b--")

IR_fit_plot_args_c = []
IR_fit_plot_kwargs_c = []

IR_plot_eff_args_c = []
IR_plot_eff_kwargs_c = []

for h_max in [2, 5, 10, 15]:
    # no oscillations, no harmonics, h can be integer
    N_h = 5  # not more needed for such simple sim
    IRASA = irasa(data=pink2, hset=np.linspace(1.1, h_max, N_h),
                  **irasa_params)
    freq_IR, _, _, IR_fit = IRASA
    IR_slope = -IR_fit["Slope"][0]

    IR_offset = 0  # set offset to 1 for alignment
    IR_offset = IR_fit["Intercept"][0]

    IR_fit = gen_aperiodic(freq_IR, (IR_offset, IR_slope))
    # label = r"$a_{IRASA}$="f"{IR_slope:.2f}, "r"$h_{max}$="f"{h_max}"
    substr = f"hmax={h_max}"
    label = fr"$a_{{{substr}}}$={IR_slope:.2f}"
    IR_kwargs = dict(label=label)
    plot_IRASA = (freq_IR, 10**IR_fit, "r--")
    IR_fit_plot_args_c.append(plot_IRASA)
    IR_fit_plot_kwargs_c.append(IR_kwargs)

    f_min = irasa_params["band"][0] / h_max
    f_max = irasa_params["band"][1] * h_max
    f_step = 1 / irasa_params["win_sec"]
    freq_IR_eff = np.arange(f_min, f_max, f_step)
    IR_fit_eff = gen_aperiodic(freq_IR_eff, (IR_offset, IR_slope))
    plot_IRASA_eff = (freq_IR_eff, 10**IR_fit_eff, "r--")

    IR_eff_kwargs = dict(alpha=0.2)
    IR_plot_eff_args_c.append(plot_IRASA_eff)
    IR_plot_eff_kwargs_c.append(IR_eff_kwargs)


# %% Plot Params

width = 7.25  # inches
panel_fontsize = 12
legend_fontsize1 = 9
legend_fontsize2 = 9
label_fontsize = 9
tick_fontsize = 9
annotation_fontsize = 9

mpl.rcParams['xtick.labelsize'] = tick_fontsize
mpl.rcParams['ytick.labelsize'] = tick_fontsize
mpl.rcParams['axes.labelsize'] = label_fontsize
mpl.rcParams['legend.fontsize'] = legend_fontsize1
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["font.size"] = 14


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

# b
xticks_b = [.1, 1, 10, 100, 600]
xlabel_b = "Frequency [Hz]"
axes_b = dict(xticks=xticks_b, xlabel=xlabel_b, xticklabels=xticks_b)

tiny_leg = dict(labelspacing=0.1, fontsize=8, borderaxespad=0, handletextpad=0,
                handlelength=2, borderpad=0)
# %% Plot

fig = plt.figure(figsize=[width, 7], constrained_layout=True)

spec = gridspec.GridSpec(4, 2, figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0])
ax3 = fig.add_subplot(spec[:2, 1])
ax4 = fig.add_subplot(spec[2:, 0])
ax5 = fig.add_subplot(spec[2:, 1])


# a
# a1
ax = ax1

# Plot sim
ax.loglog(*toy_plot_a)

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
ax.semilogx(*error_plot_a)

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

# b

ax = ax3
ax.loglog(freq_b[signal_b], psd2_noise_b[signal_b], c_sim)
ax.loglog(freq_b[noise_b], psd2_noise_b[noise_b], c_noise)
ax.loglog(*plot_fooof_b, **fooof_kwargs)
for plot_IRASA, IR_kwargs in zip(IR_fit_plot_args_b, IR_fit_plot_kwargs_b):
    ax.loglog(*plot_IRASA, **IR_kwargs)

for plot_IRASA_eff, IR_kwargs_eff in zip(IR_plot_eff_args_b,
                                         IR_plot_eff_kwargs_b):
    ax.loglog(*plot_IRASA_eff, **IR_kwargs_eff)
ax.legend(**tiny_leg)
ax.set(**axes_b)
ax.text(s="b", **abc, transform=ax.transAxes)
x_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.xaxis.set_minor_locator(x_minor)

# c

ax = ax4
ax.loglog(freq_c, psd2_noise_c, c_sim)
ax.loglog(*plot_fooof_c, **fooof_kwargs)
for plot_IRASA, IR_kwargs in zip(IR_fit_plot_args_c, IR_fit_plot_kwargs_c):
    ax.loglog(*plot_IRASA, **IR_kwargs)

for plot_IRASA_eff, IR_kwargs_eff in zip(IR_plot_eff_args_c,
                                         IR_plot_eff_kwargs_c):
    ax.loglog(*plot_IRASA_eff, **IR_kwargs_eff)

ax.legend(**tiny_leg)
ax.text(s="c", **abc, transform=ax.transAxes)
ax.set(**axes_b)
x_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.xaxis.set_minor_locator(x_minor)
plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()
