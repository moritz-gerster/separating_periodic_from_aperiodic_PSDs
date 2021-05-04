"""
Message: Fitting range borders must be 100% oscillations free.

Panel A: Oscillations crossing different fitting borders
below: fitting error. CHECK

Panel B show subj 10 huge beta:
    peak extends fitting range: 30-45Hz, 40-60Hz? 1-100Hz? 1-40?
        strong impact of delta offset: 1-100, 1-40. CHECK

(other subs:
    40-60Hz very flat, strong impact by noise: don't fit 40-60Hz if a<1?)

Pandel C:
    Show delta error, eliminate normalization

Supp. Mat. 2a): Show simulated and real time series 10 seconds
Supp. Mat. 2b): Show fooof fits
"""
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path
from fooof import FOOOF
import mne
from mne.time_frequency import psd_welch
from fooof.sim.gen import gen_aperiodic
import matplotlib as mpl
import matplotlib.gridspec as gridspec

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def noise_white(samples, seed=True):
    """Create White Noise of N samples."""
    if seed:
        np.random.seed(10)
    noise = np.random.normal(0, 1, size=samples)
    return noise


def osc_signals(samples, slopes, freq_osc=[], amp=[], width=[],
                srate=2400):
    """Simplified sim function."""
    # Initialize output
    noises = np.zeros([len(slopes), samples])
    noises_pure = np.zeros([len(slopes), samples])
    # Make fourier amplitudes
    amps = np.ones(samples//2 + 1, complex)
    freqs = np.fft.rfftfreq(samples, d=1/srate)

    # Make 1/f
    freqs[0] = 1  # avoid divison by 0
    random_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)

    for j, slope in enumerate(slopes):
        # Multiply Amp Spectrum by 1/f
        # half slope needed:
        # 1/f^2 in power spectrum = sqrt(1/f^2)=1/f^2*0.5=1/f
        # in amp spectrum
        amps = amps / freqs ** (slope / 2)
        amps *= np.exp(1j * random_phases)
        noises_pure[j] = np.fft.irfft(amps)
        for i in range(len(freq_osc)):
            # make Gaussian peak
            amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
            # normalize peak for smaller amplitude differences for different
            # frequencies:
            amp_dist /= np.max(amp_dist)
            amps += amp[i] * amp_dist
    noises[j] = np.fft.irfft(amps)
    return noises, noises_pure


# %% PARAMETERS

# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slope = [1]

# WELCH
nperseg = srate  # 4*srate too high resolution for fooof

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

c_fit1 = "purple"
c_fit2 = "turquoise"  # c
c_fit3 = "lime"
c_fit4 = "orange"

# c)

c_low = "deepskyblue"
c_med = "limegreen"
c_high = "orangered"

c_ground = "grey"


# %% A: Make signals and fit
up = 100
lower = np.arange(0, 80, 1)

freq_osc = [4, 11, 23]  # Hz
amp = [20, 20, 7]
width = [0.1, 1, 2]

signals = osc_signals(samples, slope, freq_osc, amp, width=width)
freq, noise_psd = sig.welch(signals[0][0], fs=srate, nperseg=nperseg)

# Filter 1-600Hz
freq = freq[1:601]
noise_psd = noise_psd[1:601]

# Calc fooof vary freq ranges
errors1 = []
for low in lower:
    fm = FOOOF(verbose=None)
    fm.fit(freq, noise_psd, [low, up])
    exp = fm.get_params("aperiodic", "exponent")
    error = 1 - exp
    error = np.abs(error)
    errors1.append(error)


# %% B: Load and fit
ch = 'STN_L23'

# Load data
path = "../data/Fig2/"
fname10_on = "subj10_on_R8_raw.fif"

sub10_on = mne.io.read_raw_fif(path + fname10_on, preload=True)

sub10_on.pick_channels([ch])

filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}

sub10_on.notch_filter(**filter_params)

welch_params = {"fmin": 1,
                "fmax": 600,
                "tmin": 0.5,
                "tmax": 185,
                "n_fft": srate,
                "n_overlap": srate // 2,
                "average": "mean"}

spec10_on, freq = psd_welch(sub10_on, **welch_params)
spec10_on = spec10_on[0]

frange1 = (1, 95)
frange2 = (30, 45)
frange3 = (40, 60)
frange4 = (1, 45)

fit_params = [(frange1, dict(peak_width_limits=[1, 100]), c_fit1),
              # (frange2, dict(peak_width_limits=[1, 100]), c_fit2),
              (frange2, dict(max_n_peaks=0), c_fit2),
              (frange3, dict(max_n_peaks=0), c_fit3),
              (frange4, dict(peak_width_limits=[1, 100]), c_fit4)]

fits = []
for i in range(4):
    fm = FOOOF(**fit_params[i][1])
    fm.fit(freq, spec10_on, fit_params[i][0])
    exp = fm.aperiodic_params_[1]
    ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
    fits.append((fm, exp, ap_fit))


# %% C: Reproduce PSD


# No oscillations
freq_osc = [2.5, 3,   4,    7,  27,  36,  360]
amp =      [1.5, 4.5, 5,    3, 750, 500, 6000]
width =    [0.1,  .7, 1.2, 20,   7,  11,   60]
slopes = [1]

# Make noise
w_noise = noise_white(samples)

pink1, pure1 = osc_signals(samples, slopes, freq_osc, amp, width)
pink1 = pink1[0]
pure = pure1[0]
pink1 += .0005 * w_noise
pure1 += .0005 * w_noise

freq, sim1 = sig.welch(pink1, fs=srate, nperseg=nperseg, detrend=False)
freq, pure1 = sig.welch(pure1, fs=srate, nperseg=nperseg, detrend=False)

# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
sim1 = sim1[filt]
pure1 = pure1[0, filt]

# Adjust offset for real spectrum
spec10_on_adj = spec10_on / spec10_on[-1]
sim1 /= sim1[-1]
pure1 /= pure1[-1]

# oscillations
freq_osc = [2.5, 3,   4,    7,  27,  36,  360]
amp =      [7,   4.5, 5,    3, 750, 500, 6000]
width =    [ .8,  .7, 1.2, 20,   7,  11,   60]
slopes = [1]

pink1_deltaHigh, _ = osc_signals(samples, slopes, freq_osc, amp, width)
pink1_deltaHigh = pink1_deltaHigh[0]
pink1_deltaHigh += .0005 * w_noise

freq, sim1_deltaHigh = sig.welch(pink1_deltaHigh, fs=srate, nperseg=nperseg,
                                 detrend=False)
# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
sim1_deltaHigh = sim1_deltaHigh[filt]

# Adjust offset for real spectrum
sim1_deltaHigh /= sim1_deltaHigh[-1]

# oscillations
freq_osc = [2.5, 3,  4,    7,  27,  36,  360]
amp =      [0,   0,  5,    3, 750, 500, 6000]
width =    [ .8, .7, 1.2, 20,   7,  11,   60]
slopes = [1]

pink1_deltaLow, _ = osc_signals(samples, slopes, freq_osc, amp, width)
pink1_deltaLow = pink1_deltaLow[0]
pink1_deltaLow += .0005 * w_noise

freq, sim1_deltaLow = sig.welch(pink1_deltaLow, fs=srate, nperseg=nperseg,
                                detrend=False)
# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
sim1_deltaLow = sim1_deltaLow[filt]

# Adjust offset for real spectrum
sim1_deltaLow_adj = sim1_deltaLow / sim1_deltaLow[-1]

# Fit
fm = FOOOF(**fit_params[0][1])
fm.fit(freq, spec10_on_adj, [1, 95])
exp_LFP = fm.get_params('aperiodic_params', 'exponent')
ap_fit_LFP = gen_aperiodic(fm.freqs, fm.aperiodic_params_)

fm.fit(freq, sim1_deltaLow_adj, [1, 95])
exp_low = fm.get_params('aperiodic_params', 'exponent')
ap_fit_low = gen_aperiodic(fm.freqs, fm.aperiodic_params_)

fm.fit(freq, sim1, [1, 95])
exp_med = fm.get_params('aperiodic_params', 'exponent')
ap_fit_med = gen_aperiodic(fm.freqs, fm.aperiodic_params_)

fm.fit(freq, sim1_deltaHigh, [1, 95])
exp_high = fm.get_params('aperiodic_params', 'exponent')
ap_fit_high = gen_aperiodic(fm.freqs, fm.aperiodic_params_)


# %% Plot

mpl.rcParams["font.size"] = 14

abc = dict(x=0, y=1.04, fontsize=20, fontdict=dict(fontweight="bold"))

fig = plt.figure(figsize=[9, 7.5], constrained_layout=True)

gs0 = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[10, 1.5, 10])

gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0])

ax1 = fig.add_subplot(gs00[0, 0])
ax2 = fig.add_subplot(gs00[1, 0])
ax3 = fig.add_subplot(gs00[:, 1])

gs01 = gs0[1]
ax_leg = fig.add_subplot(gs01)
ax_leg.axis("off")

gs02 = gs0[2].subgridspec(1, 3)

ax4 = fig.add_subplot(gs02[0])
ax5 = fig.add_subplot(gs02[1])
ax6 = fig.add_subplot(gs02[2])

# a)

ax = ax1
mask = (freq <= 100)
xticks = []
tick_dic = dict(xticks=xticks, xticklabels=xticks)
anno_dic = dict(x=100, ha="right", fontsize=9)
ax.loglog(freq[mask], noise_psd[mask], c_sim)
hlines_y = [8e-8, 5.58e-9, 3.9e-10]
ax.hlines(hlines_y[0], 4, 100, color=c_range1, ls="--", label="Fit range 1")
ax.hlines(hlines_y[1], 11, 100, color=c_range2, ls="--", label="Fit range 2")
ax.hlines(hlines_y[2], 23, 100, color=c_range3, ls="--",  label="Fit range 3")
ax.text(s="Fitting range: 4-100Hz", y=1.1e-7, **anno_dic)
ax.text(s="11-100Hz", y=7.1e-9, **anno_dic)
ax.text(s="23-100Hz", y=5.5e-10, **anno_dic)
ax.text(s="a", **abc, transform=ax.transAxes)
ax.set(**tick_dic, yticklabels=[],
       # xlabel="Frequency [Hz]",
       ylabel="PSD [a.u.]")

ax = ax2
ax.semilogx(lower, errors1, c_error)
ax.set(xlabel="Lower fitting range border [Hz]", ylabel="Fitting error")
ax.hlines(1, 4, 100, color=c_range1, ls="--")
ax.hlines(.7, 11, 100, color=c_range2, ls="--")
ax.hlines(.4, 23, 100, color=c_range3, ls="--")
# xticks = [1, 4, 11, 23, 100]
xticks = [1, 10, 100]
yticks = [0, 1]
tick_dic = dict(xticks=xticks, xticklabels=xticks,  yticks=yticks)
ax.set(**tick_dic)


# b)

ax = ax3
ax.loglog(freq, spec10_on, c=c_real)
for i in range(4):
    fit = fits[i][0].freqs, 10**fits[i][2], fit_params[i][2]
    freq1 = fit_params[i][0][0]
    freq2 = fit_params[i][0][1]
    if freq1 == 1:
        freq_str = f"  {freq1}-{freq2}Hz"
    else:
        freq_str = f"{freq1}-{freq2}Hz"
    kwargs = dict(lw=3, ls="--", label=freq_str + f" a={fits[i][1]:.2f}")
    ax.loglog(*fit, **kwargs)
xticks = [1, 10, 100, 600]
yticks = [5e-3, 5e-2, 5e-1]
yticklabels = [5e-3, None, .5]
ax.set(xlabel="Frequency [Hz]", xticks=xticks,
       xticklabels=xticks, yticks=yticks, yticklabels=yticklabels)
ax.set_ylabel(r"PSD [$\mu$$V^2$/Hz]", labelpad=-30)
ax.tick_params(axis="y", length=5, width=1.5)
leg = ax.legend(frameon=True, fontsize=10, bbox_to_anchor=(.55, 1.03))
leg.set_in_layout(False)
ax.text(s="b", **abc, transform=ax.transAxes)


# c)

ax = ax4

real = freq, spec10_on_adj
real_fit = fm.freqs, 10**ap_fit_LFP, "--"
real_kwargs = dict(c=c_real, alpha=.3, lw=2)

low = freq, sim1_deltaLow_adj, c_low
low_fit = fm.freqs, 10**ap_fit_low, "--"
low_kwargs = dict(c=c_low, lw=2)
x_arrow = 0.9
arr_pos_low = "", (x_arrow, 10**ap_fit_low[0]), (x_arrow, 10**ap_fit_LFP[0])

med = freq, sim1, c_med
med_fit = fm.freqs, 10**ap_fit_med, "--"
med_kwargs = dict(c=c_med, lw=2)
arr_pos_med = "", (x_arrow, 10**ap_fit_med[0]), (x_arrow, 10**ap_fit_LFP[0])

high = freq, sim1_deltaHigh, c_high
high_fit = fm.freqs, 10**ap_fit_high, "--"
high_kwargs = dict(c=c_high, lw=2)
arr_pos_high = "", (x_arrow, 10**ap_fit_high[0]), (x_arrow, 10**ap_fit_LFP[0])
# high_kwargs_k = dict(c="k", lw=2.1)

ground = freq, pure1, c_ground
ground_kwargs = dict(lw=.5)

fill_mask = freq <= 4
fill_dic = dict(alpha=0.5)
tick_dic = dict(xticks=xticks, xticklabels=xticks, yticks=[])
arrow_dic = dict(arrowprops=dict(arrowstyle="->, "
                                 "head_length=0.2,head_width=0.2", lw=2))

ax.loglog(*real, **real_kwargs, label="STN-LFP")
ax.loglog(*real_fit, **real_kwargs, label=f"fooof LFP a={exp_LFP:.2f}")
ax.loglog(*ground, **ground_kwargs, label=f"Sim 1/f a={slopes[0]}")
ax.loglog(*low)
ax.loglog(*low_fit, **low_kwargs, label=f"fooof sim1 a={exp_low:.2f}")
ax.fill_between(freq[fill_mask],
                sim1_deltaLow_adj[fill_mask], pure1[fill_mask],
                color=c_low, **fill_dic)
ax.set_ylabel("PSD [a.u.]")
ax.set(**tick_dic)
ax.annotate(*arr_pos_low, **arrow_dic)
handles, labels = ax.get_legend_handles_labels()
ax.text(s="c", **abc, transform=ax.transAxes)


ax = ax5
ax.loglog(*real, **real_kwargs)
ax.loglog(*med)
ax.loglog(*ground, **ground_kwargs)
real_line, = ax.loglog(*real_fit, **real_kwargs)
ax.loglog(*med_fit, **med_kwargs, label=f"fooof sim2 a={exp_med:.2f}")
ax.fill_between(freq[fill_mask],
                sim1[fill_mask], pure1[fill_mask], color=c_med, **fill_dic)
ax.set_yticks([], minor=True)
ax.set_xlabel("Fitting range: 1-95 Hz")
ax.spines["left"].set_visible(False)
ax.set(**tick_dic)
# ax.annotate(*arr_pos_med, **arrow_dic)
hands, labs = ax.get_legend_handles_labels()
handles.extend(hands)
labels.extend(labs)


ax = ax6
ax.loglog(*real, **real_kwargs)
# ax.loglog(*high, **high_kwargs_k)
ax.loglog(*high)
ax.loglog(*ground, **ground_kwargs)
real_line, = ax.loglog(*real_fit, **real_kwargs)
ax.loglog(*high_fit, **high_kwargs, label=f"fooof sim3 a={exp_high:.2f}")
ax.fill_between(freq[fill_mask],
                sim1_deltaHigh[fill_mask], pure1[fill_mask],
                color=c_high, **fill_dic)
ax.set_yticks([], minor=True)
ax.spines["left"].set_visible(False)
ax.set(**tick_dic)
ax.annotate(*arr_pos_high, **arrow_dic)
hands, labs = ax.get_legend_handles_labels()
handles.extend(hands)
labels.extend(labs)
leg = ax_leg.legend(handles, labels, ncol=3, frameon=True, fontsize=12,
                    labelspacing=0.1, bbox_to_anchor=(.9, .7))
leg.set_in_layout(False)
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()
# %% Plot Supp Mat b

fig, axes = plt.subplots(2, 4, figsize=[16, 8])
for i in range(4):
    ax = axes[0, i]
    kwargs = dict(add_legend=False,
                  aperiodic_kwargs=dict(color=fit_params[i][2], alpha=1),
                  data_kwargs=dict(color=c_real))
    title = f"{fit_params[i][0][0]}-{fit_params[i][0][1]}Hz"
    fits[i][0].plot(ax=ax, plt_log=True, **kwargs)
    ax.set_title(title, fontsize=30)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    if i > 0:
        ax.set_ylabel("")

    ax = axes[1, i]
    fits[i][0].plot(ax=ax, plt_log=False, **kwargs)
    if i > 0:
        ax.set_ylabel("")
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
plt.tight_layout()
plt.savefig(fig_path + fig_name[:-4] + "Supp.pdf", bbox_inches="tight")
plt.show()




# =============================================================================
# # %% TO understand:
# 
# # %% Why does oscillation disappear when using welch average=median??
# 
# # Parameters power spectrum
# freq_osc, amp, width = [100], [10000], [9]
# #freq_osc, amp, width = [100], [0000], [2]
# slopes = [2]
# 
# # Sim power spectrum
# 
# # Initialize output
# noises = np.zeros([len(slopes), samples-2])
# noises_pure = np.zeros([len(slopes), samples-2])
# # Make fourier amplitudes
# amps = np.ones(samples//2 + 1, complex)
# freqs = np.fft.rfftfreq(samples, d=1/srate)
# 
# # Make 1/f
# amps, freqs, = amps[1:], freqs[1:]  # avoid divison by 0
# # Generate random phases
# random_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)
# 
# for j, slope in enumerate(slopes):
#     # Multiply Amp Spectrum by 1/f
#     amps = amps / freqs ** (slope / 2)  # half slope needed: 1/f^2 in power spectrum = sqrt(1/f^2)=1/f^2*0.5=1/f in amp spectrum
#     amps *= np.exp(1j * random_phases)
# 
#     for i in range(len(freq_osc)):
#         # freq_idx = np.abs(freqs - freq_osc[i]).argmin() # ?????
#         # make Gaussian peak
#         amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
#         plt.plot(freqs, amp_dist)
#         plt.show()
#         plt.loglog(amps)
#         plt.ylim([1e-3, 15])
#         plt.show()
#         amp_dist /= np.max(amp_dist)    
#         noises_pure[j] = np.fft.irfft(amps)
#         amps += amp[i] * amp_dist
#         plt.loglog(amps)
#         plt.ylim([1e-3, 15])
#         plt.show()
# 
#     noises[j] = np.fft.irfft(amps)
# pink3 = noises
# 
# 
# #
# # Calc PSD
# freq_w, sim3_welch = sig.welch(pink3, fs=srate, nperseg=nperseg, average="mean")
# freq_w, sim3_pure_welch = sig.welch(noises_pure, fs=srate, nperseg=nperseg, average="mean")
# 
# filt_w = (freq_w > 0) & (freq_w <= 600)
# freq_w, sim3_welch = freq_w[filt_w], sim3_welch[0, filt_w]
# sim3_pure_welch = sim3_pure_welch[0, filt_w]
# 
# fig, axes = plt.subplots(1, 1, figsize=[8, 8])
# ax = axes
# ax.loglog(freq_w, sim3_welch, label="Sim a=3 osci")
# ax.loglog(freq_w, sim3_pure_welch, label="Sim a=3", alpha=0.5)
# #ax.set_ylim([0.00001, 1])
# ax.legend()
# plt.show()
# 
# freq_w, sim3_welch = sig.welch(pink3, fs=srate, nperseg=nperseg, average="median")
# freq_w, sim3_pure_welch = sig.welch(noises_pure, fs=srate, nperseg=nperseg, average="median")
# 
# filt_w = (freq_w > 0) & (freq_w <= 600)
# freq_w, sim3_welch = freq_w[filt_w], sim3_welch[0, filt_w]
# sim3_pure_welch = sim3_pure_welch[0, filt_w]
# 
# fig, axes = plt.subplots(1, 1, figsize=[8, 8])
# ax = axes
# ax.loglog(freq_w, sim3_welch, label="Sim a=3 osci")
# ax.loglog(freq_w, sim3_pure_welch, label="Sim a=3", alpha=0.5)
# #ax.set_ylim([0.00001, 1])
# ax.legend()
# plt.show()
# =============================================================================
