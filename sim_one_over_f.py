"""
Test different methods to get 1/f.

Test different methods to add oscillations.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from fooof import FOOOF
from scipy.stats import norm


# Signal
srate = 2400
time = np.arange(180 * srate)
n_samples = time.size
nperseg = srate  # welch

slope = 1
# freq_osc = []
# amp = []
# width = []
# random_phases=True

# Initialize output
noises = np.zeros(n_samples)
pure = np.zeros(n_samples)
# Make fourier amplitudes
amps = np.ones(n_samples//2 + 1, complex)
freqs = np.fft.rfftfreq(n_samples, d=1/srate)

# Make 1/f
freqs[0] = 1  # avoid divison by 0

amps = amps / freqs ** (slope / 2)
# random phases more realistic, but negative intereference effects
# if random_phases:
rand_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)
#rand_phases=1
amps_pure = amps * np.exp(1j * rand_phases)
# else:
  #  phase = np.pi
    #amps *= np.exp(1j * phase)
pure = np.fft.irfft(amps_pure)
# =============================================================================
# for i in range(len(freq_osc)):
#     # make Gaussian peak
#     amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
#     # normalize peak for smaller amplitude differences
#     # for different frequencies
#     amp_dist /= np.max(amp_dist)
# 
#     amps += amp[i] * amp_dist
# 
# =============================================================================
#amps *= np.exp(1j * rand_phases)
#noises[j] = np.fft.irfft(np.abs(amps))

welch_params = dict(fs=srate, nperseg=nperseg, detrend=False)

# Calc PSD
freq, pure_s = sig.welch(pure, **welch_params)

# Mask between 1Hz and 600Hz
filt = (freq > 0) & (freq <= 600)
freq = freq[filt]
pure_s = pure_s[filt]



# % C: Plot
fig, axes = plt.subplots(1, 1, figsize=[8, 8])
ax = axes
ax.loglog(freq, pure_s, label=f"1/f a={slope}")

ax.legend()
plt.show()