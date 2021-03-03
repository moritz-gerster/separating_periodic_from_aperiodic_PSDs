# %% PyEMD
import numpy as np
import scipy.signal as sig
from scipy.stats import norm
from PyEMD import EMD, EEMD, CEEMDAN
import matplotlib.pyplot as plt
from plot_spectra import load_psd


def osc_signals(samples, slope, freq_osc, amp, width=None, seed=True):
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
        # normalize
        return (noise - noise.mean()) / noise.std()
    elif isinstance(slope, (np.ndarray, list)):
        pink_noises = np.zeros([len(slope), samples-2])
        for i in range(len(slope)):
            # Multiply Amp Spectrum by 1/f
            amps_i = amps / freqs ** (slope[i] / 2)  # half slope needed
            amps_i *= np.exp(1j * random_phases)
            # Transform back to get pink noise time series
            noise = np.fft.irfft(amps_i)
            # normalize
            pink_noises[i] = (noise - noise.mean()) / noise.std()
        return pink_noises


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


# %% LOAD Real PSD
freqs, PSD_on, PSD_off = load_psd()

# select data
subj = 13
ch = 8

psd_on = PSD_on[subj, ch]
psd_off = PSD_off[subj, ch]

freq_range = [1, 45]
mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
freqs, psd_on = freqs[mask], psd_on[mask]

# %% Generate Sim PSD

# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slopes = np.arange(0, 4.5, .5)

# WELCH
win_sec = 4
nperseg = int(win_sec * srate)

freq_osc = [3, 6, 10, 20]  # Hz
amp = [1, 2, 2, 3]
width = [1, 1, 1, 5]

signals = osc_signals(samples, slopes, freq_osc, amp, width=width)
freqs_sim, psd_sim = psds_pink(signals, srate, nperseg)

freq_range = [1, 50]
mask = (freqs_sim >= freq_range[0]) & (freqs_sim <= freq_range[1])
freqs_sim, psd_sim = freqs_sim[mask], psd_sim[4, mask]


# %% EMD Parameters

emd_params = {
              # max number of iterations for each IMF.
              # Default 0 -> no max number, check for other conditions.
              "FIXE": 0,

              # min number of iterations for each IMF.
              # Default 0 -> no min number, check for other conditions.
              "FIXE_H": 0,

              # Maximum number of iterations per single sifting in EMD.
              # Default 1000.
              "MAX_ITERATION": 1000,  # computation time not an issue

              # Threshold value on energy ratio per IMF check. Default 0.2.
#              "energy_ratio_thr": 0.2,  # no impact

              # Threshold value on standard deviation per IMF check.
              # Default 0.2.
#              "std_thr": 0.2,  # no impact

              # Threshold value on scaled variance per IMF check.
              # Default 0.001.
#              "svar_thr": 0.001,  # no impact

              # Threshold value on total power per EMD decomposition.
              # Default 0.005.
              "total_power_thr": 0.005,  # bifurcation at 0.115, does not help

              # Threshold for amplitude range (after scaling) per EMD
              # decomposition. Default 0.001.
              "range_thr": 0.001,  # does not help

              # Method used to finding extrema. Default "simple".
              # Alternativ "parabol".
              "extrema_detection": "simple",  # parabol does not help

              # Data type used. Default np.float64.
              # Change to np.float16 to increase speed:
              "DTYPE": np.float64  # not an issue
              }



# %% Calc EMD and plot

emd = EMD(**emd_params)

# normalize
psd_on_norm = psd_on + psd_on.max()
psd_sim_norm = psd_sim + psd_sim.max()

# EMD
IMF = emd.emd(psd_on_norm+0, freqs)
IMF_sim = emd.emd(psd_sim_norm+0, freqs_sim)

N = IMF.shape[0]
N_sim = IMF_sim.shape[0]

# Linear fit
coef = np.polyfit(np.log10(freqs), np.log10(IMF[-1]), 1)
coef_sim = np.polyfit(np.log10(freqs_sim), np.log10(IMF_sim[-1]), 1)


fig, ax = plt.subplots(4, 2, figsize=[10, 10], sharex=False)

# Real
ax[0, 0].set_title("Real PSD")
ax[0, 0].plot(freqs, psd_on, 'r')
ax[0, 0].set_xlabel("Frequency")
ax[0, 0].set_ylabel("Power")
ax[1, 0].plot(np.log10(freqs), np.log10(psd_on), 'r')
ax[1, 0].set_xlabel("log(Frequency)")
ax[1, 0].set_ylabel("log(Power)")
ax[2, 0].plot(freqs, IMF[-1], 'g')
ax[2, 0].set_title("IMF "+str(N))
ax[2, 0].set_xlabel("Frequency")
ax[2, 0].set_ylabel("Power")
ax[3, 0].plot(np.log10(freqs), np.log10(IMF[-1]), 'g')
ax[3, 0].plot(np.log10(freqs), np.log10(freqs) * coef[0] + coef[1], 'b--',
              label=f"1/f Fit: {coef[0]:.2f}")
ax[3, 0].set_title("IMF "+str(N))
ax[3, 0].set_xlabel("log(Frequency)")
ax[3, 0].set_ylabel("log(Power)")
ax[3, 0].legend()

# Sim
ax[0, 1].set_title(f"Simulated PSD with Slope = {slopes[4]}")
ax[0, 1].plot(freqs_sim, psd_sim, 'r')
ax[0, 1].set_xlabel("Frequency")
ax[0, 1].set_ylabel("Power")
ax[1, 1].plot(np.log10(freqs_sim), np.log10(psd_sim), 'r')
ax[1, 1].set_xlabel("log(Frequency)")
ax[1, 1].set_ylabel("log(Power)")

ax[2, 1].plot(freqs_sim, IMF_sim[-1], 'g')
ax[2, 1].set_title("IMF "+str(N_sim))
ax[2, 1].set_xlabel("Frequency")
ax[2, 1].set_ylabel("Power")
ax[3, 1].plot(np.log10(freqs_sim), np.log10(IMF_sim[-1]), 'g')
ax[3, 1].plot(np.log10(freqs_sim),
              np.log10(freqs_sim) * coef_sim[0] + coef_sim[1], 'b--',
              label=f"1/f Fit: {coef_sim[0]:.2f}")
ax[3, 1].set_title("IMF "+str(N_sim))
ax[3, 1].set_xlabel("log(Frequency)")
ax[3, 1].set_ylabel("log(Power)")
ax[3, 1].legend()
plt.tight_layout()
plt.savefig("EMD_shifted.pdf")
plt.show()


# %% Calc EMD and plot

emd = EMD(**emd_params)
IMF = emd.emd(psd_on, freqs)

IMF_sim = emd.emd(psd_sim, freqs_sim)
N = max(IMF_sim.shape[0], IMF.shape[0]) + 1


fig, ax = plt.subplots(N, 2, figsize=[10, 10], sharex=True)
ax[0, 0].set_title("Real PSD")
ax[0, 0].loglog(freqs, psd_on, 'r')
ax[0, 1].set_title(f"Simulated PSD with Slope = {slopes[4]}")
ax[0, 1].loglog(freqs_sim, psd_sim, 'r')
# later: only plot last IMF
for n, imf in enumerate(IMF):
    ax[n+1, 0].loglog(freqs, imf, 'g')
    ax[n+1, 0].set_title("IMF "+str(n+1))

for n, imf in enumerate(IMF_sim):
    ax[n+1, 1].loglog(freqs_sim, imf, 'g')
    ax[n+1, 1].set_title("IMF "+str(n+1))
ax[n+1, 0].set_xlabel("Freq [Hz]")
ax[n+1, 1].set_xlabel("Freq [Hz]")
plt.tight_layout()
# plt.savefig("EMD_loglog.pdf")
plt.show()


fig, ax = plt.subplots(N, 2, figsize=[10, 10], sharex=True)
ax[0, 0].set_title("Real PSD")
ax[0, 0].plot(freqs, psd_on, 'r')
ax[0, 1].set_title(f"Simulated PSD with Slope = {slopes[4]}")
ax[0, 1].plot(freqs_sim, psd_sim, 'r')
# later: only plot last IMF
for n, imf in enumerate(IMF):
    ax[n+1, 0].plot(freqs, imf, 'g')
    ax[n+1, 0].set_title("IMF "+str(n+1))

for n, imf in enumerate(IMF_sim):
    ax[n+1, 1].plot(freqs_sim, imf, 'g')
    ax[n+1, 1].set_title("IMF "+str(n+1))
ax[n+1, 0].set_xlabel("Freq [Hz]")
ax[n+1, 1].set_xlabel("Freq [Hz]")
plt.tight_layout()
# plt.savefig("EMD.pdf")
plt.show()

# %%
emd = EMD(**emd_params)
IMF = emd.emd(psd_on, freqs)

IMF_sim = emd.emd(np.log10(psd_sim), np.log10(freqs_sim))
N = max(IMF_sim.shape[0], IMF.shape[0]) + 1


fig, ax = plt.subplots(N, 2, figsize=[10, 10], sharex=True)
ax[0, 0].set_title("Real PSD")
ax[0, 0].loglog(freqs, psd_on, 'r')
ax[0, 1].set_title(f"Simulated PSD with Slope = {slopes[4]}")
ax[0, 1].loglog(freqs_sim, psd_sim, 'r')
# later: only plot last IMF
for n, imf in enumerate(IMF):
    ax[n+1, 0].loglog(freqs, imf, 'g')
    ax[n+1, 0].set_title("IMF "+str(n+1))

for n, imf in enumerate(IMF_sim):
    ax[n+1, 1].loglog(freqs_sim, imf, 'g')
    ax[n+1, 1].set_title("IMF "+str(n+1))
ax[n+1, 0].set_xlabel("Freq [Hz]")
ax[n+1, 1].set_xlabel("Freq [Hz]")
plt.tight_layout()
# plt.savefig("EMD_loglog.pdf")
plt.show()


fig, ax = plt.subplots(N, 2, figsize=[10, 10], sharex=True)
ax[0, 0].set_title("Real PSD")
ax[0, 0].plot(freqs, psd_on, 'r')
ax[0, 1].set_title(f"Simulated PSD with Slope = {slopes[4]}")
ax[0, 1].plot(freqs_sim, psd_sim, 'r')
# later: only plot last IMF
for n, imf in enumerate(IMF):
    ax[n+1, 0].plot(freqs, imf, 'g')
    ax[n+1, 0].set_title("IMF "+str(n+1))

for n, imf in enumerate(IMF_sim):
    ax[n+1, 1].plot(freqs_sim, imf, 'g')
    ax[n+1, 1].set_title("IMF "+str(n+1))
ax[n+1, 0].set_xlabel("Freq [Hz]")
ax[n+1, 1].set_xlabel("Freq [Hz]")
plt.tight_layout()
# plt.savefig("EMD.pdf")
plt.show()







