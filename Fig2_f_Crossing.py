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


def noise_white(samples, seed=True):
    """Create White Noise of N samples."""
    if seed:
        np.random.seed(10)
    noise = np.random.normal(0, 1, size=samples)
    return noise


def psds_pink(noises, srate, nperseg, normalize=False):
    """
    Return freqs, and psds of noises array.

    Parameters
    ----------
    noises : ndarray
        Noise time series.
    srate : float
        Sample rate for Welch.
    nperseg : int
        Nperseg for Welch.
    normalize : boolean, optional
        Whether the psd amplitudes should all have a maximum of 1.
        The default is False.

    Returns
    -------
    tuple of ndarrays
        One frequency and N psd arrays of N noise signals.
    """
    noise_psds = []
    for i in range(noises.shape[0]):
        freq, psd = sig.welch(noises[i], fs=srate, nperseg=nperseg)
        # divide by max value to have the same offset at 2Hz
        if normalize:
            noise_psds.append(psd / psd.max())
        else:
            noise_psds.append(psd)
    return freq, np.array(noise_psds)


def osc_signals(samples, slope, freq_osc, amp, width=None, seed=True,
                srate=2400, normalize=True):
    """
    Generate a mixture of 1/f-aperiodic and periodic signals.

    Parameters
    ----------
    samples : int
        Number of signal samples.
    freq_osc : float or list of floats
        Peak frequencies.
    amp : float or list of floats
        Amplitudes in relation to noise.
    slope : nd.array
        1/f-slope values.
    width : None or float or list of floats, optional
        Standard deviation of Gaussian peaks. The default is None which
        corresponds to sharp delta peaks.

    Returns
    -------
    pink_noise : ndarray of size slope.size X samples
        Return signal.
    """
    if seed:
        np.random.seed(10)
    # Make fourier amplitudes
    amps = np.ones(samples//2 + 1, complex)
    freqs = np.fft.rfftfreq(samples, d=1/srate)

    # check if input is float or lists
    if isinstance(freq_osc, (int, float)):
        freq_osc = [freq_osc]  # if float, make iterable
    if isinstance(freq_osc, list):
        peaks = len(freq_osc)
    if isinstance(amp, (int, float)):
        amp = [amp] * peaks
        assert peaks == len(amp), "input lists must be of the same length"
    if isinstance(width, (int, float)):
        width = [width] * peaks
    elif isinstance(width, list):
        assert peaks == len(width), "input lists must be of the same length"


    # add Gaussian peaks to the spectrum
    if isinstance(width, list):
        for i in range(peaks):
            freq_idx = np.abs(freqs - freq_osc[i]).argmin()
            # make Gaussian peak
            if width:
                amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
                amp_dist /= np.max(amp_dist)
                amps += amp[i] * amp_dist
    # if width is none, add pure sine peaks
    elif isinstance(freq_osc, list):
        for i in range(peaks):
            freq_idx = np.abs(freqs - freq_osc[i]).argmin()
            amps[freq_idx] += amp[i]
    elif freq_osc is None:
        msg = ("what the fuck do you want? peaks or no peaks? "
               "freq_osc is None but amp is {}".format(type(amp).__name__))
        assert amp is None, msg

    amps, freqs, = amps[1:], freqs[1:]  # avoid divison by 0
    # Generate random phases
    random_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)
    if isinstance(slope, (int, float)):
        # Multiply Amp Spectrum by 1/f
        amps = amps / freqs ** (slope / 2)  # half slope needed
        amps *= np.exp(1j * random_phases)
        # Transform back to get pink noise time series
        noise = np.fft.irfft(amps)
        if normalize:
            # normalize
            return (noise - noise.mean()) / noise.std()
        else:
            return noise
    elif isinstance(slope, (np.ndarray, list)):
        pink_noises = np.zeros([len(slope), samples-2])
        for i in range(len(slope)):
            # Multiply Amp Spectrum by 1/f
            amps_i = amps / freqs ** (slope[i] / 2)  # half slope needed
            amps_i *= np.exp(1j * random_phases)
            # Transform back to get pink noise time series
            noise = np.fft.irfft(amps_i)
            if normalize:
                # normalize
                pink_noises[i] = (noise - noise.mean()) / noise.std()
            else:
                pink_noises[i] = noise
        return pink_noises


# %% PARAMETERS

# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slope = 1

# WELCH
nperseg = srate  # 4*srate too high resolution for fooof

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig2_f_crossing.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# %% A: Make signals and fit
up = 100
lower = np.arange(0, 80, 1)

freq_osc = [4, 11, 23]  # Hz
amp = [20, 20, 7]
width = [0.1, 1, 2]

signals = osc_signals(samples, slope, freq_osc, amp, width=width)
freq, noise_psd = sig.welch(signals, fs=srate, nperseg=nperseg)#, detrend=False)

# Filter 1-600Hz
freq = freq[1:601]
noise_psd = noise_psd[1:601]

# Normalize
#noise_psd /= noise_psd[0, None]

# Calc fooof vary freq ranges
errors1 = []
for low in lower:
    fm = FOOOF(verbose=None)
    fm.fit(freq, noise_psd, [low, up])
    exp = fm.get_params("aperiodic", "exponent")
    error = 1 - exp
    error = np.abs(error)
    errors1.append(error)

# %% A: Plot
fig, axes = plt.subplots(2, 1, figsize=[6, 6], sharex=False)

ax = axes[0]

mask = (freq <= 100)
ax.loglog(freq[mask], noise_psd[mask], "k",
          label=f"Power spectrum a={slope}")
ax.set_xlabel("Frequency [Hz]")

ax.hlines(5e-1, 4, 100, color="c", ls="--", label="Fit range 1")
ax.hlines(4e-1, 11, 100, color="m", ls="--", label="Fit range 2")
ax.hlines(3.3e-1, 23, 100, color="y", ls="--",  label="Fit range 3")
# =============================================================================
# ax.hlines(1e-4, 4, 100, color="c", ls="--", label="Fit range 1")
# ax.hlines(.8e-4, 11, 100, color="m", ls="--", label="Fit range 2")
# ax.hlines(.6e-4, 23, 100, color="y", ls="--",  label="Fit range 3")
# =============================================================================
xmin = ax.get_xlim()[0]
ax.text(xmin, 1, "a)", fontsize=15)
ax.legend()
ax.set_ylabel("power")

ax = axes[1]
ax.semilogx(lower, errors1, label="Fitting error")
ax.set_ylabel("Ground truth - fit")
ax.set_xlabel("Lower fitting range border")
ax.hlines(1.5, 4, 100, color="c", ls="--")
ax.hlines(1.45, 11, 100, color="m", ls="--")
ax.hlines(1.4, 23, 100, color="y", ls="--")
ax.legend(loc=2)
plt.tight_layout()
plt.show()


# %% B: Load and calc
ch = 'STN_L23'

# Load data
path = "../data/Fig2/"
fname10_off = "subj10_off_R14_raw.fif"
fname10_on = "subj10_on_R8_raw.fif"

sub10_off = mne.io.read_raw_fif(path + fname10_off, preload=True)
sub10_on = mne.io.read_raw_fif(path + fname10_on, preload=True)

sub10_off.pick_channels([ch])
sub10_on.pick_channels([ch])

filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}

sub10_off.notch_filter(**filter_params)
sub10_on.notch_filter(**filter_params)

welch_params = {"fmin": 1,
                "fmax": 600,
                "tmin": 0.5,
                "tmax": 185,
                "n_fft": srate,
                "n_overlap": srate // 2,
                "average": "mean"}

spec10_off, freq = psd_welch(sub10_off, **welch_params)
spec10_on, freq = psd_welch(sub10_on, **welch_params)

spec10_on, spec10_off = spec10_on[0], spec10_off[0]
# %% B: Fit fooof

frange1 = [1, 95]
frange2 = [30, 45]
frange3 = [40, 60]
frange4 = [1, 45]
frange5 = [1, 120]

fm1_on = FOOOF(peak_width_limits=[1, 100])
fm1_on.fit(freq, spec10_on, frange1)
ap_params1_on = fm1_on.get_params("aperiodic")
exp1_on = fm1_on.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit1_on = gen_aperiodic(fm1_on.freqs, fm1_on.aperiodic_params_)

fm1_off = FOOOF(peak_width_limits=[1, 100])
fm1_off.fit(freq, spec10_off, frange1)
ap_params1_off = fm1_off.get_params("aperiodic")
exp1_off = fm1_off.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit1_off = gen_aperiodic(fm1_off.freqs, fm1_off.aperiodic_params_)

fm2_on = FOOOF(peak_width_limits=[1, 100])
fm2_on.fit(freq, spec10_on, frange2)
ap_params1_on = fm2_on.get_params("aperiodic")
exp2_on = fm2_on.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit2_on = gen_aperiodic(fm2_on.freqs, fm2_on.aperiodic_params_)

fm2_off = FOOOF(peak_width_limits=[1, 100])
fm2_off.fit(freq, spec10_off, frange2)
ap_params1_off = fm2_off.get_params("aperiodic")
exp2_off = fm2_off.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit2_off = gen_aperiodic(fm2_off.freqs, fm2_off.aperiodic_params_)


fm3_on = FOOOF(max_n_peaks=0)
fm3_on.fit(freq, spec10_on, frange3)
ap_params1_on = fm3_on.get_params("aperiodic")
exp3_on = fm3_on.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit3_on = gen_aperiodic(fm3_on.freqs, fm3_on.aperiodic_params_)

fm3_off = FOOOF(max_n_peaks=0)
fm3_off.fit(freq, spec10_off, frange3)
ap_params1_off = fm3_off.get_params("aperiodic")
exp3_off = fm3_off.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit3_off = gen_aperiodic(fm3_off.freqs, fm3_off.aperiodic_params_)


fm4_on = FOOOF(peak_width_limits=[1, 100])
fm4_on.fit(freq, spec10_on, frange4)
ap_params1_on = fm4_on.get_params("aperiodic")
exp4_on = fm4_on.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit4_on = gen_aperiodic(fm4_on.freqs, fm4_on.aperiodic_params_)

fm4_off = FOOOF(peak_width_limits=[1, 100])
fm4_off.fit(freq, spec10_off, frange4)
ap_params1_off = fm4_off.get_params("aperiodic")
exp4_off = fm4_off.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit4_off = gen_aperiodic(fm4_off.freqs, fm4_off.aperiodic_params_)


fm5_on = FOOOF(peak_width_limits=[1, 100])
fm5_on.fit(freq, spec10_on, frange5)
ap_params1_on = fm5_on.get_params("aperiodic")
exp5_on = fm5_on.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit5_on = gen_aperiodic(fm5_on.freqs, fm5_on.aperiodic_params_)

fm5_off = FOOOF(peak_width_limits=[1, 100])
fm5_off.fit(freq, spec10_off, frange5)
ap_params1_off = fm5_off.get_params("aperiodic")
exp5_off = fm5_off.get_params("aperiodic", "exponent")
# offset = fm1.get_params("aperiodic", "offset")
ap_fit5_off = gen_aperiodic(fm5_off.freqs, fm5_off.aperiodic_params_)

# %% B: Plot

c_fit1 = "c"
c_fit2 = "orange"
c_fit3 = "g"
c_fit4 = "k"

fig, axes = plt.subplots(1, 1, figsize=[8, 8])

# ax = axes[0]
ax = axes

ax.loglog(freq, spec10_on, color="purple", label=ch + " on")
ax.loglog(fm1_on.freqs, 10**ap_fit1_on, c=c_fit1, lw=4, label=f"{frange1}Hz a={exp1_on:.2f} On")
ax.loglog(fm2_on.freqs, 10**ap_fit2_on, c=c_fit2,  lw=4, label=f"{frange2}Hz a={exp2_on:.2f} On")
ax.loglog(fm3_on.freqs, 10**ap_fit3_on, c=c_fit3,  lw=4, label=f"{frange3}Hz a={exp3_on:.2f} On")
ax.loglog(fm4_on.freqs, 10**ap_fit4_on, c=c_fit4,  lw=4, label=f"{frange4}Hz a={exp4_on:.2f} On")
# ax.loglog(fm5_on.freqs, 10**ap_fit5_on, label=f"{frange5}Hz a={exp4_on:.2f} On")
ax.legend()

# =============================================================================
# ax = axes[1]
# 
# ax.loglog(freq, spec10_off, label=ch + " Off")
# ax.loglog(fm1_off.freqs, 10**ap_fit1_off,
#           label=f"{frange1}Hz a={exp1_off:.2f} Off")
# ax.loglog(fm2_off.freqs, 10**ap_fit2_off,
#           label=f"{frange2}Hz a={exp2_off:.2f} Off")
# ax.loglog(fm3_off.freqs, 10**ap_fit3_off,
#           label=f"{frange3}Hz a={exp3_off:.2f} Off")
# ax.loglog(fm4_off.freqs, 10**ap_fit4_off,
#           label=f"{frange4}Hz a={exp4_off:.2f} Off")
# ax.loglog(fm5_off.freqs, 10**ap_fit5_off,
#           label=f"{frange5}Hz a={exp5_off:.2f} Off")
# ax.legend()
# =============================================================================
plt.show()

# %% B Supp Mat: Plot fooof

fig, axes = plt.subplots(2, 4, figsize=[16, 8])

ax = axes[0, 0]
fm1_on.plot(ax=ax, plt_log=True, add_legend=False,
            aperiodic_kwargs=dict(color=c_fit1, alpha=1),
            data_kwargs=dict(color="purple"))
ax.set_title(f"{frange1[0]}-{frange1[1]}Hz", fontsize=20)

ax = axes[1, 0]
fm1_on.plot(ax=ax, plt_log=False, add_legend=False,
            aperiodic_kwargs=dict(color=c_fit1, alpha=1),
            data_kwargs=dict(color="purple"))

ax = axes[0, 1]
fm2_on.plot(ax=ax, plt_log=True, add_legend=False,
            aperiodic_kwargs=dict(color=c_fit2, alpha=1),
            data_kwargs=dict(color="purple"))
ax.set_title(f"{frange2[0]}-{frange2[1]}Hz", fontsize=20)

ax = axes[1, 1]
fm2_on.plot(ax=ax, plt_log=False, add_legend=False,
            aperiodic_kwargs=dict(color=c_fit2, alpha=1),
            data_kwargs=dict(color="purple"))

ax = axes[0, 2]
fm3_on.plot(ax=ax, plt_log=True, add_legend=False,
            aperiodic_kwargs=dict(color=c_fit3, alpha=1),
            data_kwargs=dict(color="purple"))
ax.set_title(f"{frange3[0]}-{frange3[1]}Hz", fontsize=20)

ax = axes[1, 2]
fm3_on.plot(ax=ax, plt_log=False, add_legend=False,
            aperiodic_kwargs=dict(color=c_fit3, alpha=1),
            data_kwargs=dict(color="purple"))

ax = axes[0, 3]
fm4_on.plot(ax=ax, plt_log=True, add_legend=False,
            aperiodic_kwargs=dict(color=c_fit4, alpha=1),
            data_kwargs=dict(color="purple"))
ax.set_title(f"{frange4[0]}-{frange4[1]}Hz", fontsize=20)

ax = axes[1, 3]
fm4_on.plot(ax=ax, plt_log=False, add_legend=False,
            aperiodic_kwargs=dict(color=c_fit4, alpha=1),
            data_kwargs=dict(color="purple"))
plt.tight_layout()
plt.show()

# %% C: Reproduce PSD


def osc_signals_new(samples, slopes, freq_osc=[], amp=[], width=[],
                    srate=2400):
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
        # half slope needed: 1/f^2 in power spectrum = sqrt(1/f^2)=1/f^2*0.5=1/f
        # in amp spectrum
        amps = amps / freqs ** (slope / 2)
        amps *= np.exp(1j * random_phases)
        noises_pure[j] = np.fft.irfft(amps)
        for i in range(len(freq_osc)):
            # make Gaussian peak
            amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
            amp_dist /= np.max(amp_dist)  # normalize peak for smaller amplitude differences for different frequencies
            
            amps += amp[i] * amp_dist
    noises[j] = np.fft.irfft(amps)
    return noises, noises_pure  # delete noise_pure

#spec10_on, freq = psd_welch(sub10_on, **welch_params)
#spec10_on = spec10_on[0]

# No oscillations
freq_osc = [2.5, 3, 4, 7, 27, 36, 360]
amp =      [1.5, 4.5, 5, 3, 750, 500, 6000]
width =    [0.1, .7, 1.2, 20, 7, 11, 60]
slopes = [1]

# Make noise
w_noise = noise_white(samples)

pink1, pure1 = osc_signals_new(samples, slopes, freq_osc, amp, width)
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
freq_osc = [2.5, 3, 4, 7, 27, 36, 360]
amp =      [7, 4.5, 5, 3, 750, 500, 6000]
width =    [.8, .7, 1.2, 20, 7, 11, 60]
slopes = [1]

pink1_deltaHigh, _ = osc_signals_new(samples, slopes, freq_osc, amp, width)
pink1_deltaHigh = pink1_deltaHigh[0]
pink1_deltaHigh += .0005 * w_noise

freq, sim1_deltaHigh = sig.welch(pink1_deltaHigh, fs=srate, nperseg=nperseg, detrend=False)
# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
sim1_deltaHigh = sim1_deltaHigh[filt]

# Adjust offset for real spectrum
spec10_on_adj = spec10_on / spec10_on[-1]
sim1_deltaHigh /= sim1_deltaHigh[-1]


# oscillations
freq_osc = [2.5, 3, 4, 7, 27, 36, 360]
amp =      [0, 0, 5, 3, 750, 500, 6000]
width =    [.8, .7, 1.2, 20, 7, 11, 60]
slopes = [1]

pink1_deltaLow, _ = osc_signals_new(samples, slopes, freq_osc, amp, width)
pink1_deltaLow = pink1_deltaLow[0]
pink1_deltaLow += .0005 * w_noise

freq, sim1_deltaLow = sig.welch(pink1_deltaLow, fs=srate, nperseg=nperseg, detrend=False)
# Bandpass filter between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
sim1_deltaLow = sim1_deltaLow[filt]

# %%
# Adjust offset for real spectrum
spec10_on_adj = spec10_on / spec10_on[-1]
sim1_deltaLow_adj = sim1_deltaLow / sim1_deltaLow[-1]


fig, axes = plt.subplots(1, 1, figsize=[8, 8])
ax = axes
ax.loglog(freq, spec10_on_adj, "grey", label=ch + " on",)
ax.loglog(freq, sim1, "g", label="Sim1")
ax.loglog(freq, sim1_deltaHigh, "b", label="Sim2")
ax.loglog(freq, sim1_deltaLow_adj, "m", label="Sim3")
#ax.loglog(freq, sim3, "b", label="Sim a=3")

ax.loglog(freq, pure1, "k", lw=1, label="Sim Ground truth + noise a=1")


fm = FOOOF()
fm.fit(freq, spec10_on_adj, [1, 100])
exp_LFP = fm.get_params('aperiodic_params', 'exponent')
ap_fit_LFP = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
ax.loglog(fm.freqs, 10**ap_fit_LFP, "--", c="grey", lw=2, 
          label=f"fooof LFP a={exp_LFP:.2f}")

fm = FOOOF()
fm.fit(freq, sim1, [1, 100])
exp1 = fm.get_params('aperiodic_params', 'exponent')
ap_fit1 = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
ax.loglog(fm.freqs, 10**ap_fit1, "g--", lw=2,
          label=f"fooof sim1 a={exp1:.2f}")


fm = FOOOF()
fm.fit(freq, sim1_deltaHigh, [1, 100])
exp_high = fm.get_params('aperiodic_params', 'exponent')
ap_fit_high = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
ax.loglog(fm.freqs, 10**ap_fit_high, "b--", lw=2, 
          label=f"fooof sim2 a={exp_high:.2f}")

fm = FOOOF()
fm.fit(freq, sim1_deltaLow_adj, [1, 100])
exp_low = fm.get_params('aperiodic_params', 'exponent')
ap_fit_low = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
ax.loglog(fm.freqs, 10**ap_fit_low, "m--", lw=2, 
          label=f"fooof sim2 a={exp_low:.2f}")


ax.legend()
plt.show()


# %% Combine figures panels

fig3 = plt.figure(figsize=[10, 7.5], constrained_layout=True)
gs = fig3.add_gridspec(4, 6)

f3_ax1 = fig3.add_subplot(gs[0, 0:3])
f3_ax2 = fig3.add_subplot(gs[1, 0:3])
f3_ax3 = fig3.add_subplot(gs[:2, 3:])
f3_ax4 = fig3.add_subplot(gs[2:, :2])
f3_ax5 = fig3.add_subplot(gs[2:, 2:4])
f3_ax6 = fig3.add_subplot(gs[2:, 4:])

# a)

ax = f3_ax1
mask = (freq <= 100)
ax.loglog(freq[mask], noise_psd[mask], "k", label=f"Power spectrum a={slope}")

ax.hlines(5e-1, 4, 100, color="c", ls="--", label="Fit range 1")
ax.hlines(4e-1, 11, 100, color="m", ls="--", label="Fit range 2")
ax.hlines(3.3e-1, 23, 100, color="y", ls="--",  label="Fit range 3")
# =============================================================================
# ax.hlines(1e-4, 4, 100, color="c", ls="--", label="Fit range 1")
# ax.hlines(.8e-4, 11, 100, color="m", ls="--", label="Fit range 2")
# ax.hlines(.6e-4, 23, 100, color="y", ls="--",  label="Fit range 3")
# =============================================================================
xmin = ax.get_xlim()[0]
ax.text(0, 1, "a", fontsize=20, fontdict=dict(fontweight="bold"),
        transform=ax.transAxes)
ax.legend()
ax.set_ylabel("power")
#ax.set_ylim([1e-6, 6e-1])

plt.setp(f3_ax1.get_xticklabels(), visible=False)

ax = f3_ax2
ax.semilogx(lower, errors1, label="Fitting error")
ax.set(xlabel="Lower fitting range border", ylabel="Ground truth - fit")
ax.hlines(1.5, 4, 100, color="c", ls="--")
ax.hlines(1.45, 11, 100, color="m", ls="--")
ax.hlines(1.4, 23, 100, color="y", ls="--")
ax.legend(loc=2)

# b)

ax = f3_ax3
ax.loglog(freq, spec10_on, c="purple", label="STN LFP")
ax.loglog(fm1_on.freqs, 10**ap_fit1_on, c=c_fit1, lw=4, label=f"{frange1}Hz a={exp1_on:.2f} On")
ax.loglog(fm2_on.freqs, 10**ap_fit2_on, c=c_fit2, lw=4, label=f"{frange2}Hz a={exp2_on:.2f} On")
ax.loglog(fm3_on.freqs, 10**ap_fit3_on, c=c_fit3, lw=4, label=f"{frange3}Hz a={exp3_on:.2f} On")
ax.loglog(fm4_on.freqs, 10**ap_fit4_on, c=c_fit4, lw=4, label=f"{frange4}Hz a={exp4_on:.2f} On")
ax.legend()
ax.text(0, 1, "b", fontsize=20, fontdict=dict(fontweight="bold"),
        transform=ax.transAxes)

# c)

ax = f3_ax4
ax.loglog(freq, spec10_on_adj, "purple", label="STN LFP")
ax.loglog(freq, sim1_deltaLow_adj, c="deepskyblue", label="Sim a=1")
ax.loglog(freq, pure1, "grey", lw=1, label="Ground truth")

ax.loglog(fm.freqs, 10**ap_fit_LFP, "--", c="purple", lw=2,
          label=f"fooof LFP a={exp_LFP:.2f}")
ax.loglog(fm.freqs, 10**ap_fit_low, "--", c="deepskyblue", lw=2,
          label=f"fooof sim a={exp_low:.2f}")
ax.legend()
ax.text(0, 1, "c", fontsize=20, fontdict=dict(fontweight="bold"),
        transform=ax.transAxes)

ax = f3_ax5
ax.loglog(freq, spec10_on_adj, "purple")
ax.loglog(freq, sim1, "orange", label="Sim a=1")
ax.loglog(freq, pure1, "grey", lw=1)

ax.loglog(fm.freqs, 10**ap_fit_LFP, "--", c="purple", lw=2,
          label=f"fooof LFP a={exp_LFP:.2f}")
ax.loglog(fm.freqs, 10**ap_fit1, "--", c="orange", lw=2,
          label=f"fooof sim a={exp1:.2f}")
ax.set_yticks([])
ax.set_xlabel("Frequency [Hz]")
ax.legend()


ax = f3_ax6
ax.loglog(freq, spec10_on_adj, "purple")
ax.loglog(freq, sim1_deltaHigh, "limegreen", label="Sim a=1")
ax.loglog(freq, pure1, "grey", lw=1)

ax.loglog(fm.freqs, 10**ap_fit_LFP, "--", c="purple", lw=2,
          label=f"fooof LFP a={exp_LFP:.2f}")
ax.loglog(fm.freqs, 10**ap_fit_high, "--", c="limegreen", lw=2,
          label=f"fooof sim a={exp_high:.2f}")
ax.set_yticks([])
ax.legend()
plt.show()



"""
To do:

    good desing: whit background, spines left and bottom
    All: choose good colors
    a) eliminate legend fitting range and write next to h-lines
        
    save pdfs
    Clean up code
"""



# %%


fig, ax = plt.subplots(2, 2)

#ax[0, 0].xaxis.set_major_locator(plt.MaxNLocator(10))
#ax[0, 0].yaxis.set_major_locator(plt.MaxNLocator(1))
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.AutoLocator())
#    axi.yaxis.set_major_locator(plt.LinearLocator(5))
    axi.yaxis.set_major_locator(plt.FixedLocator([0, 0.3, 1]))
    axi.yaxis.set_major_formatter(plt.ScalarFormatter())
    axi.xaxis.set_major_formatter(plt.ScalarFormatter())





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
