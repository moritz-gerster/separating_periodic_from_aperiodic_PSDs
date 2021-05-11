"""
What: very different (realistic) 1/f exponents can lead
to the very same power sepctrum.

Why: The noise might be very strong/the noise floor very early which means
that the (very strong) oscillations hide the 1/f component.

Therefore: We cannot trust the 1/f estimates if the oscillations are on top
of the noise floor.

Which PSD: Some which have huge oscillations and probably early noise floor.
-> Since I use it in the other figures: Either LFP Sub 9 or LFP sub 10.

How 1/f: Should be 3 different exponents that look very different but all
realistic.

How oscillations: Should resemble the real spectrum a little bit.
Should be obtained like real PSD.

Pay attentions:
    - normalization: yes? how?
    - Welch: Detrend?
    - high pass filter?
    - random phases: no
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import mne
# from pathlib import Path
from fooof import FOOOF
# from mne.time_frequency import psd_welch
import scipy as sp
from numpy.fft import irfft, rfftfreq
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from fooof.sim.gen import gen_aperiodic


mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def osc_signals(slope=1, periodic_params=None, nlv=None,
                normalize=6, highpass=4, srate=2400,
                duration=180, seed=1):
    """
    Generate colored noise with optionally added oscillations.

    Parameters
    ----------
    slope : float, optional
        Aperiodic 1/f exponent. The default is 1.
    periodic_params : list of tuples, optional
        Oscillations parameters as list of tuples in form
        [(frequency, amplitude, width), (frequency, amplitude, width)].
        The default is None.
    nlv : float, optional
        Level of white noise. The default is None.
    normalize : float, optional
        Normalization factor. The default is 6.
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

    # Normalize
    if normalize:
        noise_SD = noise.std()
        scaling = noise_SD / normalize
        noise /= scaling
        noise_osc /= scaling

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


# %% PARAMETERS

# Colors
c_real = "purple"
c_sim = "k"
c_fit = "b"
c_noise = "darkgray"
c_right = "k--"
c_pure = "grey"

# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slope = 2
nperseg = srate  # welch

# Load data
# fname10 = "../data/Fig2/subj10_on_R8_raw.fif"
fname9 = "../data/Fig1/subj9_off_R1_raw.fif"

# sub10 = mne.io.read_raw_fif(fname10, preload=True)
sub9 = mne.io.read_raw_fif(fname9, preload=True)

# sub10.pick_channels(['STN_L23'])
sub9.pick_channels(['STN_R01'])

filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}

# sub10.notch_filter(**filter_params)
sub9.notch_filter(**filter_params)

# %% Calc C
# Get real data
select_times = dict(start=int(0.5*srate), stop=int(185*srate))
sub9_dat = sub9.get_data(**select_times)[0]

# Calc PSD real
welch_params = dict(fs=srate, nperseg=nperseg, detrend=False)
freq, psd_9 = sig.welch(sub9_dat, **welch_params)

filt = (freq > 0) & (freq <= 600)
freq = freq[filt]

psd_9 = psd_9[filt]

# %%

slope_s = 1
periodic_params_s = [(10.5, 4, 3),
                    (16, 2, 1.5),
                    (23, 15, 5),
                    (40, 10, 15),
                    (360, 20, 70)]

slope_m = 2
periodic_params_m = [(3, 0.3, .6),
                     (5, .4, 1),
                     (9, 3.3, 2.1),
                     (15, 4, 2.5),
                     (23, 15, 5),
                     (42, 12, 15),
                     (360, 25, 70)]

slope_l = 3
periodic_params_l = [(2, 0.07, .01),
                     (4, 1, 1.5),
                     (5, 1, 2),
                     (9.5, 4, 2),
                     (15, 5, 2.5),
                     (23, 19, 5),
                     (42, 20, 19),
                     (360, 35, 70)]

noise_s, osc_s = osc_signals(slope=slope_s,
                             periodic_params=periodic_params_s,
                             nlv=.0002, normalize=False)

noise_m, osc_m = osc_signals(slope=slope_m,
                             periodic_params=periodic_params_m,
                             nlv=.0002, normalize=False)

noise_l, osc_l = osc_signals(slope=slope_l,
                             periodic_params=periodic_params_l,
                             nlv=.00025, normalize=False)


freq, noise_psd_s = sig.welch(noise_s, **welch_params)
freq, osc_psd_s = sig.welch(osc_s, **welch_params)
freq, noise_psd_m = sig.welch(noise_m, **welch_params)
freq, osc_psd_m = sig.welch(osc_m, **welch_params)
freq, noise_psd_l = sig.welch(noise_l, **welch_params)
freq, osc_psd_l = sig.welch(osc_l, **welch_params)

freq = freq[filt]
noise_psd_s = noise_psd_s[filt]
osc_psd_s = osc_psd_s[filt]
noise_psd_m = noise_psd_m[filt]
osc_psd_m = osc_psd_m[filt]
noise_psd_l = noise_psd_l[filt]
osc_psd_l = osc_psd_l[filt]

# Normalize
norm_s = noise_psd_s[0] / psd_9[0]
norm_m = noise_psd_m[0] / psd_9[0]
norm_l = noise_psd_l[0] / psd_9[0]
noise_psd_s /= norm_s
osc_psd_s /= norm_s
noise_psd_m /= norm_m
osc_psd_m /= norm_m
noise_psd_l /= norm_l
osc_psd_l /= norm_l

# % C: Plot
fig, axes = plt.subplots(1, 1, figsize=[8, 8])
ax = axes
ax.loglog(freq, psd_9, c_real, alpha=0.4, label="LFP Sub. 9")
ax.loglog(freq, noise_psd_s, c_noise, label=f"1/f a={slope_s}")
ax.loglog(freq, osc_psd_s, c_sim, label=f"1/f a={slope_s}")
ax.loglog(freq, noise_psd_m, c_noise, label=f"1/f a={slope_m}")
ax.loglog(freq, osc_psd_m, c_sim, label=f"1/f a={slope_m}")
ax.loglog(freq, noise_psd_l, c_noise, label=f"1/f a={slope_l}")
ax.loglog(freq, osc_psd_l, c_sim, label=f"1/f a={slope_l}")
ax.set_title(periodic_params_m)
ax.legend()
# ax.set_ylim([0.004, 1.1])
# ax.set_xlim([1, 600])
plt.show()


"""

"""

# %% Fit fooof
fit_params = dict(peak_width_limits=[1, 100], verbose=False)

freq_range = [1, 95]

fm_s = FOOOF(**fit_params)
fm_m = FOOOF(**fit_params)
fm_l = FOOOF(**fit_params)
fm_real = FOOOF(**fit_params)

fm_s.fit(freq, osc_psd_s, freq_range)
fm_m.fit(freq, osc_psd_m, freq_range)
fm_l.fit(freq, osc_psd_l, freq_range)
fm_real.fit(freq, psd_9, freq_range)

exp_s = fm_s.get_params("aperiodic", "exponent")
exp_m = fm_m.get_params("aperiodic", "exponent")
exp_l = fm_l.get_params("aperiodic", "exponent")
exp_real = fm_real.get_params('aperiodic_params', 'exponent')

ap_fit_s = gen_aperiodic(fm_s.freqs, fm_s.aperiodic_params_)
ap_fit_m = gen_aperiodic(fm_m.freqs, fm_m.aperiodic_params_)
ap_fit_l = gen_aperiodic(fm_l.freqs, fm_l.aperiodic_params_)
ap_fit_real = gen_aperiodic(fm_real.freqs, fm_real.aperiodic_params_)

small_fit = fm_s.freqs, 10**ap_fit_s, "--"
med_fit = fm_m.freqs, 10**ap_fit_m, "--"
large_fit = fm_l.freqs, 10**ap_fit_l, "--"
real_fit = fm_real.freqs, 10**ap_fit_real, "--"



# %% Plot gridpec
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


real_kwargs = dict(c=c_real, alpha=.3, lw=2)
small_kwargs = dict(c=c_low, lw=2)
med_kwargs = dict(c=c_med, lw=2)
large_kwargs = dict(c=c_high, lw=2)


fill_dic = dict(alpha=0.5)
xticks = [1, 10, 100, 600]
tick_dic = dict(xticks=xticks, xticklabels=xticks, yticks=[])

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


ax = ax2

# b)

ax = ax3



# c)

ax = ax4

fill_mask = (freq > 1) & (freq <= freq_range[1])
fill_mask_fit = (fm_s.freqs > 1)


ax.loglog(freq, psd_9, **real_kwargs, label="STN-LFP")
ax.loglog(*real_fit, **real_kwargs, label=f"fooof LFP a={exp_real:.2f}")

ax.loglog(*small_fit, **small_kwargs, label=f"1/f a={exp_s:.2f}")
ax.loglog(freq, noise_psd_s, c_ground, label=f"1/f a={slope_s}")
ax.loglog(freq, osc_psd_s, **small_kwargs)

ax.fill_between(small_fit[0][fill_mask_fit], small_fit[1][fill_mask_fit],
                noise_psd_s[fill_mask],
                color=c_low, **fill_dic)
ax.set(**tick_dic)
handles, labels = ax.get_legend_handles_labels()
ax.text(s="c", **abc, transform=ax.transAxes)



ax = ax5
ax.loglog(freq, psd_9, **real_kwargs)
ax.loglog(*real_fit, **real_kwargs)

ax.loglog(*med_fit, **med_kwargs, label=f"1/f a={exp_m:.2f}")
ax.loglog(freq, noise_psd_m, c_ground, label=f"1/f a={slope_m}")
ax.loglog(freq, osc_psd_m, **med_kwargs)

ax.fill_between(med_fit[0][fill_mask_fit], med_fit[1][fill_mask_fit],
                noise_psd_m[fill_mask],
                color=c_med, **fill_dic)
ax.set(**tick_dic)
ax.set_yticks([], minor=True)
hands, labs = ax.get_legend_handles_labels()
handles.extend(hands)
labels.extend(labs)
ax.set_xlabel("Fitting range: 1-95 Hz")
ax.spines["left"].set_visible(False)


ax = ax6
ax.loglog(freq, psd_9, **real_kwargs)

ax.loglog(*large_fit, **large_kwargs, label=f"1/f a={exp_l:.2f}")
ax.loglog(freq, noise_psd_l, c_ground, label=f"1/f a={slope_l}")
ax.loglog(*real_fit, **real_kwargs)
ax.loglog(freq, osc_psd_l, **large_kwargs)

ax.fill_between(large_fit[0][fill_mask_fit], large_fit[1][fill_mask_fit],
                noise_psd_l[fill_mask],
                color=c_high, **fill_dic)
ax.set_yticks([], minor=True)
ax.set(**tick_dic)
ax.spines["left"].set_visible(False)
hands, labs = ax.get_legend_handles_labels()
handles.extend(hands)
labels.extend(labs)
leg = ax_leg.legend(handles, labels, ncol=4, frameon=True, fontsize=12,
                    labelspacing=0.1, bbox_to_anchor=(.9, .7))
leg.set_in_layout(False)
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.show()


# %% PLOT SINGLE
c_low = "darkorange"

fig, axes = plt.subplots(1, 1, figsize=[5, 5])

ax = axes

fill_mask = (freq > 1) & (freq <= freq_range[1])
fill_mask_fit = (fm_s.freqs > 1)


ax.loglog(freq, psd_9, **real_kwargs, label="STN-LFP")
ax.loglog(*real_fit, **real_kwargs, label=f"fooof LFP a={exp_real:.2f}")

ax.loglog(*small_fit, **small_kwargs, label=f"1/f a={exp_s:.2f}")
ax.loglog(freq, noise_psd_s, c_low, lw=0.5, label=f"1/f a={slope_s}")
ax.loglog(freq, osc_psd_s, **small_kwargs)

# =============================================================================
# ax.fill_between(small_fit[0][fill_mask_fit], noise_psd_m[fill_mask],
#                 noise_psd_s[fill_mask],
#                 color=c_high, **fill_dic)
# =============================================================================
ax.set(**tick_dic)
handles, labels = ax.get_legend_handles_labels()
ax.text(s="c", **abc, transform=ax.transAxes)


ax.loglog(freq, psd_9, **real_kwargs)
ax.loglog(*real_fit, **real_kwargs)

ax.loglog(*med_fit, **med_kwargs, label=f"1/f a={exp_m:.2f}")
ax.loglog(freq, noise_psd_m, c_med, lw=0.5, label=f"1/f a={slope_m}")
ax.loglog(freq, osc_psd_m, **med_kwargs)

ax.set(**tick_dic)
ax.set_yticks([], minor=True)
hands, labs = ax.get_legend_handles_labels()
handles.extend(hands)
labels.extend(labs)
ax.set_xlabel("Fitting range: 1-95 Hz")
ax.spines["left"].set_visible(False)


# =============================================================================
# ax = ax6
# ax.loglog(freq, psd_9, **real_kwargs)
# 
# ax.loglog(*large_fit, **large_kwargs, label=f"1/f a={exp_l:.2f}")
# ax.loglog(freq, noise_psd_l, c_ground, label=f"1/f a={slope_l}")
# ax.loglog(*real_fit, **real_kwargs)
# ax.loglog(freq, osc_psd_l, **large_kwargs)
# 
# ax.fill_between(large_fit[0][fill_mask_fit], large_fit[1][fill_mask_fit],
#                 noise_psd_l[fill_mask],
#                 color=c_high, **fill_dic)
# ax.set_yticks([], minor=True)
# ax.set(**tick_dic)
# ax.spines["left"].set_visible(False)
# hands, labs = ax.get_legend_handles_labels()
# handles.extend(hands)
# labels.extend(labs)
# =============================================================================
