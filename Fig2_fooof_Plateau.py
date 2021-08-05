"""Figure 1."""
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


def osc_signals(slope, periodic_params=None, nlv=None,
                highpass=4, srate=2400,
                duration=180, seed=1):
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
    normalize : float, optional
        Normalization factor. The default is 6.
    highpass : int, optional
        The order of the butterworth highpass filter. The default is 4. If
        None, no filter will be applied.
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
        sos = sig.butter(highpass, 1, btype="hp", fs=srate, output='sos')
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


# %% Parameters

# Colors
# a)
c_sim = "k"
c_fits = ["limegreen", "r", "r", "r"]  # c="dodgerblue"
c_noise = "darkgray"
c_ground_a = "k"

# b)
c_real = "purple"
# c_low = "limegreen"

# c)
c_low = "deepskyblue"
c_med = "limegreen"
c_high = "#ff7f00"


c_ground = "grey"

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig2_Plateau.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# %% Load data

# Litvak file names
path = "../data/Fig1/"
fname9 = "subj9_off_R1_raw.fif"

sub9 = mne.io.read_raw_fif(path + fname9, preload=True)

# Select channel
sub9.pick_channels(['STN_R01'])

# Filter out line noise
filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}

sub9.notch_filter(**filter_params)

# Convert mne to numpy
srate = 2400
start = int(0.5 * srate)  # artefact in beginning of recording
stop = int(185 * srate)  # artefact at the end of recording

sub9 = sub9.get_data(start=start, stop=stop)[0]

# Calc Welch
welch_params = dict(fs=srate, nperseg=srate)  # detrend=False?


freq, psd_sub9 = sig.welch(sub9, **welch_params)

# Mask above highpass and below lowpass
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
psd_sub9 = psd_sub9[filt]

# %% Calc a)

# Make noise
slope_a = 2
noise_params_a = dict(slope=slope_a, nlv=0.00005, highpass=False, seed=3)
pink2, _ = osc_signals(**noise_params_a)

# Calc PSD
freq, psd2_noise = sig.welch(pink2, **welch_params)

# Mask above highpass and below lowpass
freq, psd2_noise = freq[filt], psd2_noise[filt]

# Normalize
psd2_noise /= psd2_noise[0]

# Detect Noise floor
floor_a = detect_noise_floor(freq, psd2_noise, f_start=1)
signal_a = (freq <= floor_a)
noise_a = (freq >= floor_a)

ground_truth = gen_aperiodic(freq, np.array([0, slope_a]))

# %% Plot params a)
plot_sim = (freq[signal_a], psd2_noise[signal_a], c_sim)
plot_plateau = (freq[noise_a], psd2_noise[noise_a], c_noise)
plot_ground = (freq, 10**ground_truth, c_ground_a)

# plot limits, ticks, and labels
xlim_a = (1, 600)
y_plateau = psd2_noise[freq == xlim_a[1]][0]
ylim_a = (.4*y_plateau, 1)
xticks_a = [1, 10, 50, 100, 200, 600]
ylabel_a = "PSD [a.u.]"
axes_a = dict(xlim=xlim_a, ylim=ylim_a, xticks=xticks_a, xticklabels=xticks_a,
              ylabel=ylabel_a)

# fit for different upper fit limits and plot
plot_fits, dic_fits, vlines = [], [], []

upper_fit_limits = xticks_a[1:-1]
fit_alphas = [.9, 0.6, .4, .2]

fm = FOOOF(verbose=False)
for i, lim in enumerate(upper_fit_limits):
    fm.fit(freq, psd2_noise, [1, lim])
    exp = fm.get_params('aperiodic_params', 'exponent')
    fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
    label = fr"1-{lim}Hz $\beta=${exp:.2f}"
    plot_fit = fm.freqs, 10**fit, "-"
    dic_fit = dict(c=c_fits[i], lw=2,
                   # alpha=fit_alphas[i],
                   label=label)
    # annotate x-crossing
    vline = lim, ylim_a[0], 10**fit[-1]
    plot_fits.append(plot_fit)
    dic_fits.append(dic_fit)
    vlines.append(vline)
dic_line = dict(color=c_sim, linestyle=":", lw=.3)

# %% Plot params b)

# Detect Noise floor
floor_9 = detect_noise_floor(freq, psd_sub9, f_start=1)

# Mask signal/noise
signal_9 = (freq <= floor_9)

noise_9 = (freq >= floor_9)

# Prepare plot
plot_sub9 = (freq, psd_sub9, c_real)
plot_sub9_signal = (freq[signal_9], psd_sub9[signal_9], c_real)

plot_sub9_plateau = (freq[noise_9], psd_sub9[noise_9], c_noise)

# Get Oscillation coordinates sub9
peak_freq1 = 23
peak_freq2 = 350

peak_height1 = psd_sub9[freq == peak_freq1]
peak_height2 = psd_sub9[freq == peak_freq2]

noise_height_osc = psd_sub9[freq == floor_9]

# Create lines, arrows, and text to annotate noise floor
line_osc1 = dict(x=peak_freq1, ymin=noise_height_osc*0.8, ymax=peak_height1,
                 color=c_sim, linestyle="--", lw=.5)

line_osc2 = dict(x=peak_freq2, ymin=noise_height_osc*0.8, ymax=peak_height2,
                 color=c_sim, linestyle="--", lw=.5)

arrow1 = dict(text="",
              xy=(floor_9, noise_height_osc*0.8),
              xytext=(peak_freq1, noise_height_osc*0.8),
              arrowprops=dict(arrowstyle="->", color=c_sim, lw=1))
arrow2 = dict(text="",
              xy=(floor_9, noise_height_osc*0.8),
              xytext=(peak_freq2, noise_height_osc*0.8),
              arrowprops=dict(arrowstyle="->", color=c_sim, lw=1))

plateau_line9 = dict(text="",
                     xy=(floor_9, noise_height_osc*0.86),
                     xytext=(floor_9, noise_height_osc*.5),
                     arrowprops=dict(arrowstyle="-", color=c_sim, lw=2))


plateau_txt9 = dict(text=f"{floor_9}Hz",
                    xy=(floor_9, noise_height_osc*0.7),
                    xytext=(floor_9*1.02, noise_height_osc*.53))

xticks_b = [1, 10, 100, 600]
xlim_b = (1, 826)
ylim_b = (0.005, 2)
ylabel_b = r"PSD [$\mu$$V^2$/Hz]"

axes_b = dict(xlabel=None, ylabel=ylabel_b, xlim=xlim_b, ylim=ylim_b,
              xticks=xticks_b, xticklabels=xticks_b)


# %% Calc c)

# Simulate psd_sub9
slope1 = 1
slope15 = 1.5
slope2 = 2

osc_params1 = [(3, 0, 1),
               (5, 0, 1),
               (10.5, 4, 3),
               (16, 2, 1.5),
               (23, 15, 5),
               (42, 9, 15),
               (360, 25, 80)]

osc_params15 = [(3, 0, 1),
                (5, .1, 1),
                (10.5, 4, 3),
                (16, 2, 1.5),
                (23, 15, 5),
                (42, 15, 22),
                (360, 30, 80)]

osc_params2 = [(3, 0.1, .6),
               (5, .3, 1),
               (10.5, 4.1, 3), # (9, 2.8, 2), (10.5, 3, 3),
               (16, 2.1, 2), # (15, 3.9, 2.5), (16, 5, 4),
               (23, 14, 5),
               (42, 15, 20),
               (360, 25, 70)]

noise1, osc1 = osc_signals(slope1,
                           periodic_params=osc_params1,
                           nlv=.0002)

noise15, osc15 = osc_signals(slope15,
                             periodic_params=osc_params15,
                             nlv=.0002)

noise2, osc2 = osc_signals(slope2,
                           periodic_params=osc_params2,
                           nlv=.0002)

# Calc Welch
freq, psd_noise1 = sig.welch(noise1, **welch_params)
freq, psd_noise15 = sig.welch(noise15, **welch_params)
freq, psd_noise2 = sig.welch(noise2, **welch_params)
freq, psc_osc1 = sig.welch(osc1, **welch_params)
freq, psc_osc15 = sig.welch(osc15, **welch_params)
freq, psc_osc2 = sig.welch(osc2, **welch_params)

# Filter above highpass and below lowpass
freq = freq[filt]
psd_noise1 = psd_noise1[filt]
psd_noise15 = psd_noise15[filt]
psd_noise2 = psd_noise2[filt]
psc_osc1 = psc_osc1[filt]
psc_osc15 = psc_osc15[filt]
psc_osc2 = psc_osc2[filt]

# Normalize
norm1 = psd_noise1[0] / psd_sub9[0]
norm15 = psd_noise15[0] / psd_sub9[0]
norm2 = psd_noise2[0] / psd_sub9[0]
psd_noise1 /= norm1
psc_osc1 /= norm1
psd_noise15 /= norm15
psc_osc15 /= norm15
psd_noise2 /= norm2
psc_osc2 /= norm2

# Fit fooof
freq_range = [1, 95]  # upper border above oscillations range, below plateau
fooof_params = dict(peak_width_limits=[1, 100], verbose=False)

fm1 = FOOOF(**fooof_params)
fm15 = FOOOF(**fooof_params)
fm2 = FOOOF(**fooof_params)
fm_sub9 = FOOOF(**fooof_params)

fm1.fit(freq, psc_osc1, freq_range)
fm15.fit(freq, psc_osc15, freq_range)
fm2.fit(freq, psc_osc2, freq_range)
fm_sub9.fit(freq, psd_sub9, freq_range)

# Extract fit results
exp1 = fm1.get_params("aperiodic", "exponent")
exp15 = fm15.get_params("aperiodic", "exponent")
exp2 = fm2.get_params("aperiodic", "exponent")
exp_sub9 = fm_sub9.get_params('aperiodic_params', 'exponent')
off_sub9 = fm_sub9.get_params('aperiodic_params', 'offset')

# Simulate fitting results
ap_fit1 = gen_aperiodic(fm1.freqs, fm1.aperiodic_params_)
ap_fit15 = gen_aperiodic(fm15.freqs, fm15.aperiodic_params_)
ap_fit2 = gen_aperiodic(fm2.freqs, fm2.aperiodic_params_)
ap_fit_sub9 = gen_aperiodic(fm_sub9.freqs, fm_sub9.aperiodic_params_)

fit1 = fm1.freqs, 10**ap_fit1, c_low
fit15 = fm15.freqs, 10**ap_fit15, c_med
fit2 = fm2.freqs, 10**ap_fit2, c_low
fit_sub9 = fm_sub9.freqs, 10**ap_fit_sub9, c_real

psd_plateau_fits = [fit1, fit15, fit2]

spec9_fit_label = fr"fooof LFP $\beta=${exp_sub9:.2f}"

# % Plot params c)

plot_sub9 = freq, psd_sub9, c_real

plot_osc1 = freq, psc_osc1, c_low
plot_osc15 = freq, psc_osc15, c_med
plot_osc2 = freq, psc_osc2, c_high

# Summarize
psd_plateau_vary = [plot_osc1, plot_osc15, plot_osc2]

plot_noise1 = freq, psd_noise1, c_ground
plot_noise15 = freq, psd_noise15, c_ground
plot_noise2 = freq, psd_noise2, c_ground

# ground1 = gen_aperiodic(freq, np.array([off_sub9, slope1]))
# ground15 = gen_aperiodic(freq, np.array([off_sub9, slope15]))
# ground2 = gen_aperiodic(freq, np.array([off_sub9, slope2]))

# plot_ground1 = freq, 10**ground1, c_low
# plot_ground15 = freq, 10**ground15, c_med
# plot_ground2 = freq, 10**ground2, c_high

xlim_c = (1, 825)
xlabel_c = "Frequency in Hz"

low_kwargs = dict(c=c_low, ls="-", lw=2, alpha=1)
med_kwargs = dict(c=c_med, ls="-", lw=2, alpha=1)
high_kwargs = dict(c=c_high, ls="-", lw=2, alpha=1)

# Summarize
plateau_kwargs = [low_kwargs, med_kwargs, high_kwargs]
plateau_labels = [r"$\beta_{fit}$="f"{exp1:.2f}",
                  r"$\beta_{fit}$="f"{exp15:.2f}",
                  r"$\beta_{fit}$="f"{exp2:.2f}"]
plateau_labels = [fr"fooof flat $\beta=${exp1:.2f}",
                  fr"fooof med $\beta=${exp15:.2f}",
                  fr"fooof steep $\beta=${exp2:.2f}"]
psd_aperiodic_vary = [plot_noise1, plot_noise15, plot_noise2]

labelpad = 5
leg_c = dict(ncol=3, loc=10, bbox_to_anchor=(.54, -.3), borderpad=0.35)
axes_c = dict(xticks=xticks_b, xticklabels=xticks_b, xlim=xlim_c)

noise_power = (freq > 1) & (freq <= freq_range[1])
freq_mask = freq[noise_power]
plot_delta_low = (freq_mask, psd_noise1[noise_power],
                  10**ap_fit1[fm1.freqs > 1])
plot_delta_med = (freq_mask, psd_noise15[noise_power],
                  10**ap_fit15[fm1.freqs > 1])
plot_delta_high = (freq_mask, psd_noise2[noise_power],
                   10**ap_fit2[fm1.freqs > 1])

# Summarize
delta_power = [plot_delta_low, plot_delta_med, plot_delta_high]
colors_c = [c_low, c_med, c_high]
# % Plot Params

width = 7.25  # inches
panel_fontsize = 12
legend_fontsize = 9
label_fontsize = 9
tick_fontsize = 9
annotation_fontsize = tick_fontsize

mpl.rcParams['xtick.labelsize'] = tick_fontsize
mpl.rcParams['ytick.labelsize'] = tick_fontsize
mpl.rcParams['axes.labelsize'] = label_fontsize
mpl.rcParams['legend.fontsize'] = legend_fontsize
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

panel_labels = dict(x=0, y=1.01, fontsize=panel_fontsize,
                    fontdict=dict(fontweight="bold"))

line_fit = dict(lw=2, ls=":", zorder=5)
line_ground = dict(lw=.5, ls="--", zorder=5)
psd_aperiodic_kwargs = dict(lw=0.5)


# %% Plot

# Prepare Gridspec
fig = plt.figure(figsize=[width, 6.5], constrained_layout=True)

gs0 = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[10, 1, 10])

# a) and b)
gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0])
ax1 = fig.add_subplot(gs00[0])
ax2 = fig.add_subplot(gs00[1])

# Legend suplot
gs01 = gs0[1]
ax_leg = fig.add_subplot(gs01)
ax_leg.axis("off")

# c)
gs02 = gs0[2].subgridspec(1, 3)
ax3 = fig.add_subplot(gs02[0])
ax4 = fig.add_subplot(gs02[1])
ax5 = fig.add_subplot(gs02[2])

c_axes = [ax3, ax4, ax5]


# a) =========================================================================
ax = ax1

# Plot simulated PSD and ground truth
ax.loglog(*plot_sim, label=fr"1/f $\beta=${slope_a} + noise")
ax.loglog(*plot_ground, **line_ground, label=fr"Ground truth $\beta=${slope_a}")

# Plot plateau in grey
ax.loglog(*plot_plateau, label="Plateau")

# Plot fits for different upper fitting borders
for i in range(len(upper_fit_limits)):
    ax.loglog(*plot_fits[i], **dic_fits[i])
    ax.vlines(*vlines[i], **dic_line)


# Set axes
ax.set(**axes_a)
ax.legend()
ax.text(s="a", **panel_labels, transform=ax.transAxes)
# =============================================================================


# b)  =========================================================================
ax = ax2

# Plot Sub 9
ax.loglog(*plot_sub9_signal, label="LFP")

#   ax.loglog(*fit_sub9, **line_fit, label=spec9_fit_label)

ax.loglog(*plot_sub9_plateau, label="Plateau")

# Plot Peak lines
ax.vlines(**line_osc1)
ax.vlines(**line_osc2)

# Plot Arrow left and right
ax.annotate(**arrow1)
ax.annotate(**arrow2)

# Annotate noise floor start as line
ax.annotate(**plateau_line9)

# Annotate noise floor start as text
ax.annotate(**plateau_txt9, fontsize=annotation_fontsize)

# Set axes
ax.set(**axes_b)
ax.legend(loc=0)
ax.text(s="b", **panel_labels, transform=ax.transAxes)

# =============================================================================

# c)

# Make sure we have just one label for each repetitive plot
spec9_label = ["STN-LFP", None, None]
spec9_fit_labels = [spec9_fit_label, None, None]
aperiodic_label = [None, None, "1/f + noise"]

for i, ax in enumerate(c_axes):

    # Plot LFP and fooof fit
    ax.loglog(*plot_sub9, alpha=0.3, lw=2, label=spec9_label[i])
    ax.loglog(*fit_sub9, **line_fit, label=spec9_fit_labels[i])

    # Plot sim low delta power and fooof fit
    ax.loglog(*psd_plateau_vary[i])
    ax.loglog(*psd_plateau_fits[i], **plateau_kwargs[i],
              label=plateau_labels[i])

    # Plot aperiodic component of sim
    ax.loglog(*psd_aperiodic_vary[i], **psd_aperiodic_kwargs,
              label=aperiodic_label[i])

    # Indicate delta power as fill between aperiodic component
    # and full spectrum
    ax.fill_between(*delta_power[i], color=colors_c[i], alpha=0.2)

    # Draw arrow
    if i != 1:
        # ax.annotate(**arrows[i])
        pass
    else:
        ax.set_xlabel(xlabel_c)

    # Save legend handles labels and set axes
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.set_ylabel(ylabel_a, labelpad=labelpad)
        ax.text(s="c", **panel_labels, transform=ax.transAxes)
    else:
        hands, labs = ax.get_legend_handles_labels()
        handles.extend(hands)
        labels.extend(labs)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([], minor=True)
        ax.set_yticks([])
    ax.set(**axes_c)


# Set legend between subplots
leg = ax_leg.legend(handles, labels, **leg_c)
leg.set_in_layout(False)

plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()
