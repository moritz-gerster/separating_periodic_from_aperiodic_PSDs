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
from pathlib import Path
from fooof import FOOOF
from mne.time_frequency import psd_welch
from scipy.stats import norm
import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def noise_white(samples, seed=True):
    """Create White Noise of N samples."""
    if seed:
        np.random.seed(10)
    noise = np.random.normal(0, 1, size=samples)
    return noise


def osc_signals_new(samples, slopes, freq_osc=[], amp=[], width=[],
                    srate=2400, random_phases=False):
    """Return osc signals New."""
    if isinstance(slopes, (float, int)):
        slopes = [slopes]
    # Initialize output
    noises = np.zeros([len(slopes), samples])
    noises_pure = np.zeros([len(slopes), samples])
    # Make fourier amplitudes
    amps = np.ones(samples//2 + 1, complex)
    freqs = np.fft.rfftfreq(samples, d=1/srate)

    # Make 1/f
    freqs[0] = 1  # avoid divison by 0

    for j, slope in enumerate(slopes):
        # Multiply Amp Spectrum by 1/f
        # half slope needed:
        # 1/f^2 in power spectrum = sqrt(1/f^2)=1/f^2*0.5=1/f
        # in amp spectrum
        amps = amps / freqs ** (slope / 2)
        # random phases more realistic, but negative intereference effects
        if random_phases:
            rand_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)
            amps *= np.exp(1j * rand_phases)
        else:
            phase = 0
            amps *= np.exp(1j * phase)
        noises_pure[j] = np.fft.irfft(amps)
        for i in range(len(freq_osc)):
            # make Gaussian peak
            amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
            # normalize peak for smaller amplitude differences
            # for different frequencies
            amp_dist /= np.max(amp_dist)

            amps += amp[i] * amp_dist
    noises[j] = np.fft.irfft(amps)
    return noises, noises_pure  # delete noise_pure



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
fname10 = "../data/Fig2/subj10_on_R8_raw.fif"
fname9 = "../data/Fig1/subj9_off_R1_raw.fif"

sub10 = mne.io.read_raw_fif(fname10, preload=True)
sub9 = mne.io.read_raw_fif(fname9, preload=True)

sub10.pick_channels(['STN_L23'])
sub9.pick_channels(['STN_R01'])

filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}

sub10.notch_filter(**filter_params)
sub9.notch_filter(**filter_params)

# %% Calc C

slope_m = 1
freq_osc_m = [0]  # [ 1, 6,   8,  23,  55, 360]
amp_m =      [0]  # [ 0, 0,  0, 0, 0, 0]
width_m =    [0]  # [.8, 2, 1.9,   6,  10, 50]

# Gen signal
oscs_m = slope_m, freq_osc_m, amp_m, width_m
pink_m, pure_m = osc_signals_new(samples, *oscs_m)

# Add white noise
w_noise = noise_white(samples)
pink_m += .05 * w_noise
pure_m += .05 * w_noise

# Highpass filter
sos = sig.butter(20, 1, btype="hp", fs=srate, output='sos')
pink_m = sig.sosfilt(sos, pink_m)[0]
pure_m = sig.sosfilt(sos, pure_m)[0]

# Get real data
select_times = dict(start=int(0.5*srate), stop=int(185*srate))
sub9_dat = sub9.get_data(**select_times)[0]
sub10_dat = sub10.get_data(**select_times)[0]

# Calc PSD real
welch_params = dict(fs=srate, nperseg=nperseg, detrend=False)
freq, psd_9 = sig.welch(sub9_dat, **welch_params)
freq, psd_10 = sig.welch(sub10_dat, **welch_params)

# Calc PSD sim
freq, sim_m = sig.welch(pink_m, **welch_params)
freq, pure_m = sig.welch(pure_m, **welch_params)

# Mask between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]

psd_9 = psd_9[filt]
psd_10 = psd_10[filt]
sim_m = sim_m[filt]
pure_m = pure_m[filt]

# Adjust offset for real spectrum
#psd_9_norm /= psd_9_norm[-1]
#sim_m /= sim_m[-1]
#pure_m /= pure_m[-1]

# % C: Plot
fig, axes = plt.subplots(1, 1, figsize=[8, 8])
ax = axes
ax.loglog(freq, psd_9_norm, c_real, alpha=0.4, label="LFP Sub. 9")
ax.loglog(freq, sim_m, c_sim, label=f"Sim a={slope_m}")
ax.loglog(freq, pure_m, c_noise, label=f"1/f a={slope_m}")
ax.set_title(f"Freqs: {freq_osc_m}\nAmps: {amp_m}\nWidths: {width_m}")
ax.legend()
plt.show()