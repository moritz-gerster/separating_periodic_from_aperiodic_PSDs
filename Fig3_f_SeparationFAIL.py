"""
IDEA: show that fooof cannot handle overlapping peaks.

Does not work: if peaks crossing fitting range is avoided, fooof works.

I cannot explain/reproduce why fooof fails so badly for the absence seizure.
"""
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from fooof import FOOOF
from scipy.stats import norm
from fooof.sim.gen import gen_aperiodic


def osc_signals(samples, slopes, freq_osc=[], amp=[], width=[],
                srate=2400, seed=1):
    """Simplified sim function."""
    if seed:
        np.random.seed(seed)
    # Initialize output
    slopes = [slopes]
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


def calc_PSDs(seiz_data, seiz_len, srate, **welch_params):
    """Calc PSDs."""
    # n_freq = srate // 2 + 1
    nperseg = welch_params["nperseg"]
    n_freq = np.fft.rfftfreq(nperseg, d=1/srate).size

    psd_pre = np.zeros(n_freq)
    psd_seiz = np.zeros(n_freq)
    psd_post = np.zeros(n_freq)

    data_pre = seiz_data[:10*srate]
    data_seiz = seiz_data[10*srate:10*srate + seiz_len]
    data_post = seiz_data[10*srate + seiz_len:30*srate]

    freq, psd_pre = sig.welch(data_pre, **welch_params)
    freq, psd_seiz = sig.welch(data_seiz, **welch_params)
    freq, psd_post = sig.welch(data_post, **welch_params)

    return freq, psd_pre, psd_seiz, psd_post


# %% Parameters a)

# Colors
c1 = "darkorange"
c2 = "g"
c12 = "k"

# Sim
duration = 180
srate_sim = 2400
time = np.arange(duration * srate_sim)
samples = time.size

# Welch
nperseg = srate_sim
welch_params_a = {"fs": srate_sim, "nperseg": nperseg}

# 1/f
slope_flat = 0
slope = 1

# Osc1
freq_osc1 = [5, 10, 20, 30, 40]
amp1 = [15, 0, 15, 15, 0]
width1 = [.1, 1.2, 1.2, 1.2, 1.2]

# Osc2
freq_osc2 = [7.5, 15, 25, 35]
amp2 = [15, 15, 15, 15]
width2 = [.7, 1.2, 1.2, 1.2]

# Osc1 + Osc2
freq_osc12 = freq_osc1 + freq_osc2
amp12 = amp1 + amp2
width12 = width1 + width2

# Tuples
osc1_tup = freq_osc1, amp1, width1
osc2_tup = freq_osc2, amp2, width2
osc12_tup = freq_osc12, amp12, width12

# Fooof
freq_range = (1, 45)
fooof_params_a = dict(peak_width_limits=(0, 12), verbose=False)
# %% a)

# Generate Oscillations
osc1, _ = osc_signals(samples, slope_flat, *osc1_tup)
osc2, _ = osc_signals(samples, slope_flat, *osc2_tup)
osc12, _ = osc_signals(samples, slope_flat, *osc12_tup)

noise_osc1, _ = osc_signals(samples, slope, *osc1_tup)
noise_osc2, _ = osc_signals(samples, slope, *osc2_tup)
noise_osc12, _ = osc_signals(samples, slope, *osc12_tup)

# Calc PSDS
freq, osc1_psd = sig.welch(osc1, **welch_params_a)
freq, osc2_psd = sig.welch(osc2, **welch_params_a)
freq, osc12_psd = sig.welch(osc12, **welch_params_a)
freq, noise_osc1_psd = sig.welch(noise_osc1, **welch_params_a)
freq, noise_osc2_psd = sig.welch(noise_osc2, **welch_params_a)
freq, noise_osc12_psd = sig.welch(noise_osc12, **welch_params_a)

# Mask
mask = (freq > 1) & (freq <= 50)
freq = freq[mask]
osc1_psd = osc1_psd[0][mask]
osc2_psd = osc2_psd[0][mask]
osc12_psd = osc12_psd[0][mask]
noise_osc1_psd = noise_osc1_psd[0][mask]
noise_osc2_psd = noise_osc2_psd[0][mask]
noise_osc12_psd = noise_osc12_psd[0][mask]

# Normalize for same starting point
noise_osc1_psd /= noise_osc1_psd[0]
noise_osc2_psd /= noise_osc2_psd[0]
noise_osc12_psd /= noise_osc12_psd[0]

# Set exactly equal endpoint
noise_osc1_psd[-1] = noise_osc12_psd[-1]
noise_osc2_psd[-1] = noise_osc12_psd[-1]

# Fit
fm1 = FOOOF(**fooof_params_a)
fm1.fit(freq, noise_osc1_psd, freq_range)
fm1_fit = gen_aperiodic(fm1.freqs, fm1.aperiodic_params_)
a1 = fm1.aperiodic_params_[1]

fm2 = FOOOF(**fooof_params_a)
fm2.fit(freq, noise_osc2_psd, freq_range)
fm2_fit = gen_aperiodic(fm2.freqs, fm2.aperiodic_params_)
a2 = fm2.aperiodic_params_[1]

fm12 = FOOOF(**fooof_params_a)
fm12.fit(freq, noise_osc12_psd, freq_range)
fm12_fit = gen_aperiodic(fm12.freqs, fm12.aperiodic_params_)
a12 = fm12.aperiodic_params_[1]

# Arrow
x_arrow = 2
arr_pos = "", (x_arrow, 10**fm12_fit[0]), (x_arrow, 10**fm1_fit[0])
arrow_dic = dict(arrowprops=dict(arrowstyle="->, "
                                 "head_length=0.2,head_width=0.2", lw=2))
fooof_plot = dict(plot_peaks="shade", plt_log=True, add_legend=False)
# % Plot a)

fig, axes = plt.subplots(3, 3, figsize=[15, 10], sharey="row")#,
#                         gridspec_kw=dict(width_ratios=[1, .5, .5]))
ax = axes[0, 0]
ax.loglog(freq, osc1_psd, c1, label="Osc1: 6, 12, 20Hz")
ax = axes[0, 1]
ax.loglog(freq, osc2_psd, c2, label="Osc2: 3, 9, 15, 25Hz")
ax = axes[0, 2]
ax.loglog(freq, osc12_psd, c12, label="Osc1+Osc2")
ax.legend()

ax = axes[1, 0]
ax.loglog(freq, noise_osc1_psd, c1, label="1/f noise osc1 a=1")
#ax.loglog(fm1.freqs, 10**fm1_fit, c1,
 #         label=f"1/f noise osc1 fooof fit a={a1:.2f}")

ax = axes[1, 0]
ax.loglog(freq, noise_osc2_psd, c2, label="1/f noise osc1+osc2 a=1")
#ax.loglog(fm2.freqs, 10**fm2_fit, c=c2,
 #         label=f"1/f noise osc1+osc2 fooof fit a={a12:.2f}")

ax = axes[1, 0]
ax.loglog(freq, noise_osc12_psd, c12, label="1/f noise osc1+osc2 a=1")

ax = axes[1, 1]
ax.loglog(fm12.freqs, 10**fm12_fit, c=c12,
          label=f"1/f noise osc1+osc2 fooof fit a={a12:.2f}")
ax.loglog(fm1.freqs, 10**fm1_fit, c1,
          label=f"1/f noise osc1 fooof fit a={a1:.2f}")
ax.loglog(fm2.freqs, 10**fm2_fit, c=c2,
          label=f"1/f noise osc2 fooof fit a={a2:.2f}")
ax.annotate(*arr_pos, **arrow_dic)
ax.legend()
ax = axes[2, 0]
fm1.plot(ax=ax, **fooof_plot)
ax = axes[2, 1]
fm2.plot(ax=ax, **fooof_plot)
ax = axes[2, 2]
fm12.plot(ax=ax, **fooof_plot)

plt.show()