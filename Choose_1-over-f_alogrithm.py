"""
Test different methods to get 1/f.

Test different methods to add oscillations.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from fooof import FOOOF
from scipy.stats import norm
from fooof.sim.gen import gen_aperiodic
from numpy.fft import irfft, rfftfreq, rfft

def powerlaw_psd_gaussian(exponent, size, fmin=0):
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1./samples) # Low frequency cutoff
    ix   = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)

    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2. # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(None,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    sr = np.random.normal(scale=s_scale, size=size)
    si = np.random.normal(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2): si[...,-1] = 0

    # Regardless of signal length, the DC component must be real
    si[...,0] = 0

    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma

    return y


# %%
# Signal
srate = 2400
time = np.arange(180 * srate)
n_samples = time.size
nperseg = srate  # welch

slope = .5

# Initialize output
amps = np.ones(n_samples//2, complex)
freqs = rfftfreq(n_samples, d=1/srate)
#freqs[0] = 1  # avoid divison by 0
freqs = freqs[1:]  # avoid divison by 0


# Non random amps
# amps_nonRand = amps / freqs ** (slope / 2)
# phase = 1

# First random then pink:
rand_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)
#rand_phases_real = np.random.uniform(0, 2*np.pi, size=amps.shape)
#rand_phases_imag = np.random.uniform(0, 2*np.pi, size=amps.shape)
amps_rand = amps * np.exp(1j * rand_phases)

# =============================================================================
# plt.plot(amps_rand[:5].imag)
# plt.plot(amps_rand[:5].real)
# plt.plot(np.abs(amps_rand[:5]))
# plt.plot(np.sqrt(amps_rand[:5].real**2 + amps_rand[:5].imag**2))
# plt.show()
# =============================================================================

amps_rand_pink = amps_rand / freqs ** (slope / 2)

# =============================================================================
# plt.plot(amps_rand_pink[:500].imag)
# plt.plot(amps_rand_pink[:500].real)
# plt.plot(np.abs(amps_rand_pink[:500]))
# plt.show()
# 
# plt.loglog(np.abs(amps_rand_pink)[:50])
# plt.loglog(amps_rand_pink[:50])
# # plt.loglog(amps_nonRand[:50])
# plt.show()
# =============================================================================

# Make inverse fft
# noise_nonRand = irfft(amps_nonRand)
noise_rand_pink = irfft(amps_rand_pink)
# noise_rand_pink_real = ifft(amps_rand_pink.real)
# noise_rand_pink_abs = irfft(np.abs(amps_rand_pink))

# np.allclose(noise_nonRand, noise_rand_pink_abs)# True
#plt.plot(noise_rand_pink[100*srate:102*srate])
#plt.plot(noise_rand_pink_abs[100*srate:102*srate])

# =============================================================================
# White noise method
w_noise = np.random.normal(0, 1, size=n_samples-1)
amps_white = rfft(w_noise)
amps_white_pink = amps_white / freqs ** (slope / 2)
noise_white_pink = irfft(amps_white_pink)
# noise_white_pink_real = irfft(amps_white_pink.real)
noise_white_pink_abs = irfft(np.abs(amps_white_pink))
# =============================================================================

# =============================================================================
# plt.loglog(amps_white[:1000])
# plt.loglog(amps_white.imag[:1000])
# plt.loglog(np.abs(amps_white)[:1000])
# =============================================================================

noise_github = powerlaw_psd_gaussian(slope, n_samples)
noise_github2 = powerlaw_psd_gaussian(2*slope, n_samples)
noise_github3 = powerlaw_psd_gaussian(3*slope, n_samples)


# Highpass filter
sos = sig.butter(4, 1, btype="hp", fs=srate, output='sos')
# noise_nonRand = sig.sosfilt(sos, noise_nonRand)
noise_rand_pink = sig.sosfilt(sos, noise_rand_pink)
noise_white_pink = sig.sosfilt(sos, noise_white_pink)
# noise_rand_pink_real = sig.sosfilt(sos, noise_rand_pink_real)
# noise_white_pink_real = sig.sosfilt(sos, noise_white_pink_real)
# noise_rand_pink_abs = sig.sosfilt(sos, noise_rand_pink_abs)
noise_white_pink_abs = sig.sosfilt(sos, noise_white_pink_abs)
noise_github = sig.sosfilt(sos, noise_github)
noise_github2 = sig.sosfilt(sos, noise_github2)
noise_github3 = sig.sosfilt(sos, noise_github3)

# np.allclose(noise_nonRand, noise_rand_pink_abs)# True

# plt.plot(noise_rand_pink_abs[srate:2*srate])
# plt.plot(noise_nonRand[srate:2*srate])


# Plot time series
# =============================================================================
# plt.plot(noise_white_pink[15*srate:20*srate])
# plt.plot(noise_rand_pink[15*srate:20*srate])
# plt.show()
# =============================================================================

# Calc PSD
welch_params = dict(fs=srate, nperseg=nperseg, detrend="l")
# freq, noise_nonRand_psd = sig.welch(noise_nonRand, **welch_params)
freq, noise_rand_pink_psd = sig.welch(noise_rand_pink, **welch_params)
freq, noise_white_pink_psd = sig.welch(noise_white_pink, **welch_params)
# freq, noise_rand_pink_real_psd = sig.welch(noise_rand_pink_real, **welch_params)
# freq, noise_white_pink_real_psd = sig.welch(noise_white_pink_real, **welch_params)
# freq, noise_rand_pink_abs_psd = sig.welch(noise_rand_pink_abs, **welch_params)
freq, noise_white_pink_abs_psd = sig.welch(noise_white_pink_abs, **welch_params)

freq, noise_github_psd = sig.welch(noise_github, **welch_params)
freq, noise_github_psd2 = sig.welch(noise_github2, **welch_params)
freq, noise_github_psd3 = sig.welch(noise_github3, **welch_params)

# np.allclose(noise_nonRand_psd, noise_rand_pink_abs_psd)# True

# Mask between 1Hz and 600Hz
#filt = (freq > 0) & (freq <= 600)
#freq = freq[filt]
#noise_psd = noise_psd[filt]


# noise_nonRand_psd /= noise_nonRand_psd[1]
noise_rand_pink_psd /= noise_rand_pink_psd[1]
noise_white_pink_psd /= noise_white_pink_psd[1]
# noise_rand_pink_real_psd /= noise_rand_pink_real_psd[1]
# noise_white_pink_real_psd /= noise_white_pink_real_psd[1]
# noise_rand_pink_abs_psd /= noise_rand_pink_abs_psd[1]
noise_white_pink_abs_psd /= noise_white_pink_abs_psd[1]
noise_github_psd /= noise_github_psd[1]

ap_fooof = gen_aperiodic(freq, np.array([0, slope]))

# np.allclose(noise_nonRand_psd, noise_rand_pink_abs_psd)# True


# % C: Plot
fig, axes = plt.subplots(1, 1, figsize=[8, 8])
ax = axes
# ax.loglog(freq, noise_nonRand_psd, "--", label="Non random phases")
ax.loglog(freq, noise_rand_pink_psd, lw=4, label="noise_rand_pink_psd")
ax.loglog(freq, noise_white_pink_psd, label="noise_white_pink_psd")
# ax.loglog(freq, noise_rand_pink_real_psd, label="noise_rand_pink_real_psd")
# ax.loglog(freq, noise_white_pink_real_psd, label="noise_white_pink_real_psd")
# ax.loglog(freq, noise_rand_pink_abs_psd, label="noise_rand_pink_abs_psd")
ax.loglog(freq, noise_white_pink_abs_psd, label="noise_white_pink_abs_psd")
ax.loglog(freq, noise_github_psd, label="noise_github_psd")
# ax.loglog(freq, noise_github_psd2, label="noise_github_psd2")
# ax.loglog(freq, noise_github_psd3, label="noise_github_psd3")


ax.loglog(freq, 10**ap_fooof, label="fooof")

ax.set_xlim([1, 600])
ax.legend()
plt.show()

"""
1. lesson: I need random phases to create a realistic signal.
2. lesson: taking the real part of complex amplitudes yields the same as
taking the imaginary part: random phases very the amplitudes randomly over real
and imaginary axis.
3. lesson: taking the real part of complex amplitudes yields the same as
taking the complex amplitdues. The REAL inverse fourier transformation ignores
the imaginary part.
4. lesson: taking the absolute complex amplitdues yields the same as taking
the amplitdues of non-random phases because sqrt(real**2+imag**2) is alway 1.

4. Taking the absolute complex fourier amplitudes of white noise is different
from taking the real fourier amplitudes of white noise. The offset at 2Hz
is lower but I don't know why. In the case of white noise it is allowed to take
the absolute amplitdues.
5. The projeffional git hub version produces the same output as
noise_rand_pink and noise_white_psd.
-> I will trust these and not noise_white_pink_abs.

Since noise_rand_pink is the simplest algorithm I will use this one.
"""


