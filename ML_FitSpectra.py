"""
Use linear regression to find best fitting parameters.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import mne
from pathlib import Path
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
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


# %%

# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slope = 2
nperseg = srate  # welch

w_noise = noise_white(samples)

# Load data
path = "../data/Fig1/"
fname9 = "subj9_off_R1_raw.fif"

sub9 = mne.io.read_raw_fif(path + fname9, preload=True)

sub9.pick_channels(['STN_R01'])

filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": .5,
                 "method": "spectrum_fit"}

sub9.notch_filter(**filter_params)

welch_params = {"fmin": 1,
                "fmax": 600,
                "tmin": 0.5,
                "tmax": 185,
                "n_fft": srate,
                "n_overlap": srate // 2,
                "average": "mean"}


spec9, freq = psd_welch(sub9, **welch_params)

psd_lfp_osc = spec9[0]

c_sim = "k"
c_noise = "darkgray"
c_real = "purple"

# %%


def sim_psd(psd_lfp_osc, **parameters):

    freq_osc_m = parameters["freq_osc"]
    amp_m = parameters["amp"]
    width_m = parameters["width"]
    slope_m = parameters["slope"]

    oscs_m = slope_m, freq_osc_m, amp_m, width_m

    pink_m, pure_m = osc_signals_new(samples, *oscs_m)

    pink_m = pink_m[0]
    pure_m = pure_m[0]

    pink_m += parameters["noise_scale"] * w_noise
    pure_m += parameters["noise_scale"] * w_noise

    freq, sim_m = sig.welch(pink_m, fs=srate, nperseg=nperseg, detrend=False)

    # Bandpass filter between 1Hz and 600Hz
    filt = (freq > 0) & (freq <= 600)
    freq = freq[filt]
    sim_m = sim_m[filt]

    # Adjust offset for real spectrum
    psd_lfp_osc /= psd_lfp_osc[0]
    sim_m /= sim_m[0]

    minimize = (psd_lfp_osc - sim_m).sum()
    return minimize, parameters


# parameter to fit:
parameters = dict(n_freq=1, freq_osc=[23], amp=[1], width=[5], slope=1,
                  noise_scale=0.0003)



sim_psd(psd_lfp_osc, **parameters)

# %% C: Plot
freq, pure_m = sig.welch(pure_m, fs=srate, nperseg=nperseg, detrend=False)
pure_m = pure_m[filt]
pure_m /= pure_m[0]


fig, axes = plt.subplots(1, 1, figsize=[8, 8])
ax = axes

ax.loglog(freq, psd_lfp_osc, c_real, alpha=0.4, label="LFP Sub. 9")
ax.loglog(freq, sim_m, c_sim, label=f"Sim a={slope_m}")
ax.loglog(freq, pure_m, c_noise, label=f"1/f a={slope_m}")

ax.legend()
plt.show()

