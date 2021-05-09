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
                    srate=2400, random_phases=True):
    """Return osc signals New."""
    slopes = [slopes] if isinstance(slopes, (float, int)) else slopes

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
            #rand_phases=1
            amps_pure = amps * np.exp(1j * rand_phases)
        else:
            phase = np.pi
            #amps *= np.exp(1j * phase)
        noises_pure[j] = np.fft.irfft(amps_pure)
        for i in range(len(freq_osc)):
            # make Gaussian peak
            amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
            # normalize peak for smaller amplitude differences
            # for different frequencies
            amp_dist /= np.max(amp_dist)

            amps += amp[i] * amp_dist
        print(amps)
    amps *= np.exp(1j * rand_phases)
    noises[j] = np.fft.irfft(np.abs(amps))
    return noises, noises_pure  # delete noise_pure



# %% Test function
# =============================================================================
# srate_lowN = 10000
# 
# time_lowN = np.arange(180 * srate_lowN)
# samples_lowN = time_lowN.size
# nperseg_lowN = srate_lowN  # welch
# 
# slope_s = .01
# slope_m = .5
# slope_l = 1
# 
# # Gen signal
# pink_s, pure_s = osc_signals_new(samples_lowN, slope_s, srate=srate_lowN, random_phases=True)
# _, pure_m = osc_signals_new(samples_lowN, slope_m, srate=srate_lowN)
# pink_l, pure_l = osc_signals_new(samples_lowN, slope_l, srate=srate_lowN, random_phases=True)
# 
# # Highpass filter
# # sos = sig.butter(20, 1, btype="hp", fs=srate, output='sos')
# # pink_s = sig.sosfilt(sos, pink_s)
# # pure_m = sig.sosfilt(sos, pure_m)
# 
# # Add white noise
# # w_noise = noise_white(samples)
# # pure_s += .00035 * w_noise
# # pure_m += .05 * w_noise
# 
# # Calc PSD
# welch_low = dict(fs=srate_lowN, nperseg=nperseg_lowN)
# freq_large, pure_s = sig.welch(pure_s, **welch_low)
# freq_large, pure_m = sig.welch(pure_m, **welch_low)
# freq_large, pure_l = sig.welch(pure_l, **welch_low)
# 
# freq_large, pink_s = sig.welch(pink_s, **welch_low)
# freq_large, pink_l = sig.welch(pink_l, **welch_low)
# 
# pure_l = pure_l[0]
# pure_s = pure_s[0]
# 
# fooof_s = gen_aperiodic(freq_large, np.array([0, slope_s]))
# fooof_l = gen_aperiodic(freq_large, np.array([0, slope_l]))
# 
# fm = FOOOF()
# 
# fm.fit(freq_large, pure_l, freq_range=[1, 100])
# exp = fm.get_params('aperiodic_params', 'exponent')
# fooof_l_fit = gen_aperiodic(freq_large, np.array([0, exp]))
# 
# 
# fig, ax = plt.subplots(1, 1)
# ax.loglog(freq_large, pure_s, label="func s")
# ax.loglog(freq_large, pure_l, label="func l")
# ax.loglog(freq_large, 10**fooof_s, label="Fooof s")
# ax.loglog(freq_large, 10**fooof_l, label="Fooof l")
# ax.loglog(freq_large, 10**fooof_l_fit, label="Fooof l fit")
# ax.legend()
# =============================================================================

# =============================================================================
# # Mask between 1Hz and 600Hz
# filt = (freq_large <= 5000)
# freq_large = freq_large[filt]
# 
# # Normalize
# pure_s = pure_s[0] / pure_s[0][0]
# pure_m = pure_m[0] / pure_m[0][0]
# pure_l = pure_l[0] / pure_l[0][0]
# 
# pink_s = pink_s[0] / pink_s[0][0]
# pink_l = pink_l[0] / pink_l[0][0]
# 
# pure_s = pure_s[filt]
# pure_m = pure_m[filt]
# pure_l = pure_l[filt]
# 
# pink_s = pink_s[filt]
# pink_l = pink_l[filt]
# =============================================================================


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

slope_s = 0.6
freq_osc_s = [ 1, 6,   8,  23,  55, 360]
amp_s =      np.array([ 3, 4,  19, 300, 150, 3000]) * 2
width_s =    [.8, 2, 1.9,   6,  10, 50]

slope_m = 1
freq_osc_m = [0]  # [ 1, 6,   8,  23,  55, 360]
amp_m =      [0]  # [ 3, 4,  19, 300, 150, 2000]
width_m =    [0]  # [.8, 2, 1.9,   6,  10, 50]

slope_l = 1.4
freq_osc_l = [0]  # [ 1, 6,   8,  23,  55, 360]
amp_l =      [0]  #[ 3, 4,  19, 300, 150, 2000]
width_l =    [0]  # [.8, 2, 1.9,   6,  10, 50]

# Gen signal
oscs_s = slope_s, freq_osc_s, amp_s, width_s
oscs_m = slope_m, freq_osc_m, amp_m, width_m
oscs_l = slope_l, freq_osc_l, amp_l, width_l

pink_s, pure_s = osc_signals_new(samples, *oscs_s, random_phases=True)
pink_m, pure_m = osc_signals_new(samples, *oscs_m)
pink_l, pure_l = osc_signals_new(samples, *oscs_l)



# Highpass filter
#sos = sig.butter(20, 1, btype="hp", fs=srate, output='sos')
#pink_m = sig.sosfilt(sos, pink_m)[0]
#pure_m = sig.sosfilt(sos, pure_m)[0]
#pure_s = sig.sosfilt(sos, pure_s)[0]
pure_s = pure_s[0]
pure_m = pure_m[0]
pure_l = pure_l[0]

pink_s = pink_s[0]
pink_m = pink_m[0]
pink_l = pink_l[0]


# Add white noise
w_noise = noise_white(samples)
#pure_s += 2e-4 * w_noise
pink_s += 2e-4 * w_noise
pure_m += .0005 * w_noise
pink_m += .0005 * w_noise
pure_l += .001 * w_noise
pink_l += .001 * w_noise

# Get real data
select_times = dict(start=int(0.5*srate), stop=int(185*srate))
sub9_dat = sub9.get_data(**select_times)[0]
# sub10_dat = sub10.get_data(**select_times)[0]

# Calc PSD real
welch_params = dict(fs=srate, nperseg=nperseg, detrend=False)
freq, psd_9 = sig.welch(sub9_dat, **welch_params)
# freq, psd_10 = sig.welch(sub10_dat, **welch_params)

# Calc PSD sim
#freq, sim_m = sig.welch(pink_m, **welch_params)
freq, pure_s = sig.welch(pure_s, **welch_params)
freq, pure_m = sig.welch(pure_m, **welch_params)
freq, pure_l = sig.welch(pure_l, **welch_params)

freq, pink_s = sig.welch(pink_s, **welch_params)
freq, pink_m = sig.welch(pink_m, **welch_params)
freq, pink_l = sig.welch(pink_l, **welch_params)


# Mask between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]

psd_9 = psd_9[filt]
# psd_10 = psd_10[filt]
#sim_m = sim_m[filt]
pure_s = pure_s[filt]
pure_m = pure_m[filt]
pure_l = pure_l[filt]

pink_s = pink_s[filt]
pink_m = pink_m[filt]
pink_l = pink_l[filt]


# Scale PSD (a.u.)

# Adjust offset for real spectrum
# =============================================================================
# psd_9 /= psd_9[0]
# 
# pure_s /= pure_s[0]
# pure_m /= pure_m[0]
# pure_l /= pure_l[0]
# 
# pink_s /= pink_s[0]
# pink_m /= pink_m[0]
# pink_l /= pink_l[0]
# =============================================================================


# % C: Plot
fig, axes = plt.subplots(1, 1, figsize=[8, 8])
ax = axes
ax.loglog(freq, psd_9, c_real, alpha=0.4, label="LFP Sub. 9")
#ax.loglog(freq, psd_10, c_real, alpha=0.4, label="LFP Sub. 10")
# ax.loglog(freq, sim_m, c_sim, label=f"Sim a={slope_m}")
ax.loglog(freq, pure_s, c_noise, label=f"1/f a={slope_s}")
#ax.loglog(freq, pure_m, c_noise, label=f"1/f a={slope_m}")
#ax.loglog(freq, pure_l, c_noise, label=f"1/f a={slope_l}")

ax.loglog(freq, pink_s, c_sim, label=f"1/f a={slope_s}")
#ax.loglog(freq, pink_m, c_noise, label=f"1/f a={slope_m}")
#ax.loglog(freq, pink_l, c_noise, label=f"1/f a={slope_l}")
ax.set_title(f"Freqs: {freq_osc_m}\nAmps: {amp_m}\nWidths: {width_m}")
ax.legend()
plt.show()