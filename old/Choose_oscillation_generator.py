"""
I decided on a correct 1/f algorithm. Now I need to add oscillations while
avoiding destructive interference.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from fooof.sim.gen import gen_power_spectrum
from numpy.fft import irfft, rfftfreq
from scipy.stats import norm


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
            amp_dist = norm(freq_osc, width).pdf(freqs)
            amp_dist /= np.max(amp_dist)  # scale
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

    # Normalize
    if normalize:
        noise_SD = noise.std()
        scaling = noise_SD / normalize
        noise /= scaling
        noise_osc /= scaling

    # Highpass filter
    if highpass:
        sos = sig.butter(highpass, 1, btype="hp", fs=srate, output='sos')
        noise = sig.sosfilt(sos, noise)
        noise_osc = sig.sosfilt(sos, noise_osc)

    return noise, noise_osc


srate = 2400
slope = 1
periodic_params = [[3, 1, 1], [5, 1, 1], [10, 1, 1], [50, 1, 5], [52, 1, 5]]

# fooof
aperiodic_params = (0, slope)
freq_range = [1, 600]
f_fooof, fooof = gen_power_spectrum(freq_range, aperiodic_params,
                                    periodic_params,
                                    freq_res=1, nlv=0.005)

# Own function
noise_func, osc_func = osc_signals(slope=slope,
                                   periodic_params=periodic_params,
                                   nlv=0.0001)

# Calc PSD
welch_params = dict(fs=srate, nperseg=srate, detrend="l")
freq, noise_func_psd = sig.welch(noise_func, **welch_params)
freq, osc_func_psd = sig.welch(osc_func, **welch_params)

# % C: Plot
fig, axes = plt.subplots(1, 1, figsize=[8, 8])
ax = axes
ax.loglog(freq, osc_func_psd, lw=3, label="own func")
ax.loglog(freq, noise_func_psd, lw=3, label="own func")
ax.loglog(f_fooof, fooof, label="fooof osc")
ax.set_xlim(freq_range)
ax.legend()
plt.show()

# =============================================================================
# # %% Make time series from fooof:
# f_fooof, fooof = gen_power_spectrum(freq_range, aperiodic_params,
#                                     periodic_params,
#                                     freq_res=0.001, nlv=0.5)
#
# noise_fooof = irfft(np.sqrt(fooof))
# plt.plot(noise_fooof[srate:2*srate])
# =============================================================================
