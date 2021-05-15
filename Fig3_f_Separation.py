"""Fooof needs clearly separable (and ideally Gaussian) peaks."""
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
from fooof import FOOOF
from scipy.stats import norm
from fooof.sim.gen import gen_aperiodic
import matplotlib as mpl


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


def saw_noise(srate, seiz_len, slope=[2], saw_power=0.9,
              saw_width=0.54, freq=3, duration=30, seed=1, **welch_params):
    """Make Docstring."""
    if seed:
        np.random.seed(seed)
    # duration and sampling
    time = np.arange(duration * srate)
    samples = time.size

    # Sawtooth signal
    t = np.arange(0, duration, 1/srate)
    saw = sawtooth(2 * np.pi * freq * t, width=saw_width)
    saw = saw[:-2]
    saw *= saw_power  # scaling

    # 1/f noise
    noises, _ = osc_signals(samples, slope)
    noises = noises[0]

    # Highpass 0.3 Hz like real signal
    # sos = sig.butter(20, 1, btype="hp", fs=srate, output='sos')
    # noises = sig.sosfilt(sos, noises)

    # make signal 10 seconds zero, 10 seconds strong, 10 seconds zero
    saw_seiz = np.r_[np.zeros(pre),
                     saw[:seiz_len],
                     np.zeros(post)]

    noise_saw = noises + saw_seiz

    # normalize
    noise_saw = (noise_saw - noise_saw.mean()) / noise_saw.std()

    # PSD
    freq, saw_pre = sig.welch(noise_saw[:pre], **welch_params)
    freq, saw_seiz = sig.welch(noise_saw[pre:pre+seiz_len], **welch_params)
    freq, saw_post = sig.welch(noise_saw[pre + seiz_len:], **welch_params)
    return t, noise_saw, freq, saw_pre, saw_seiz, saw_post


def ap_fit_label(psd: np.array, cond: str, freq: np.array,
                 freq_range: tuple,
                 fooof_params: dict) -> tuple:
    """
    Return aperiodic fit and corresponding label.

    Parameters
    ----------
    psd : np.array
        PSD.
    cond : str
        Condition.
    freq : np.array
        Freq array for PSD.
    freq_range : tuple of int
        Fitting range.
    fooof_params : dict
        Fooof params.

    Returns
    -------
    tuple(ndarray, str)
        (aperiodic fit, plot label).
    """
    fm = FOOOF(**fooof_params)
    fm.fit(freq, psd, freq_range)
    exp = fm.get_params("aperiodic", "exponent")
    label = f"1/f {cond}={exp:.2f}"
    ap_fit = gen_aperiodic(freq, fm.aperiodic_params_)
    return 10**ap_fit, label


# %% Parameters

# Colors
c_empirical = "purple"
c_sim = "k"

c_seiz = "r"
c_pre = "c"
c_post = "y"

srate_ep = 256
seiz_start = 87800
seiz_end = 91150
seiz_len = seiz_end - seiz_start

pre = 10 * srate_ep
post = 20 * srate_ep - seiz_len

pre_seiz = seiz_start - pre
post_seiz = pre_seiz + 3 * pre

cha_nm = "F3-C3"

nperseg = srate_ep
saw_power = 0.004
saw_width = 0.69
seed = 2
slope = 1.8
seiz = 0
welch_params_c = {"fs": srate_ep, "nperseg": nperseg}
saw_params = dict(slope=slope, saw_power=saw_power, saw_width=saw_width,
                  seed=seed)
fooof_params = dict(verbose=False)  # standard params

# Paths
data_path = "../data/Fig3/"
fig_path = "../paper_figures/"
fig_name = "Fig3_f_Separation.pdf"
fig_name_supp = "Fig3_f_Separation_SuppMat.pdf"

# %% Get data

seiz_data = np.load(data_path + cha_nm + ".npy", allow_pickle=True)

# normalize
seiz_data = seiz_data[pre_seiz:post_seiz]
# seiz_data /= seiz_data.std()

t, noise_saw, freq, saw_pre, saw_seiz, saw_post = saw_noise(srate_ep, seiz_len,
                                                            **saw_params)

freq, psd_pre, psd_seiz, psd_post = calc_PSDs(seiz_data, seiz_len, srate_ep,
                                              **welch_params_c)


# %% Fit

freq_range = [1, 100]

fooof_inp = dict(freq=freq, freq_range=freq_range, fooof_params=fooof_params)

ap_pre_eeg, lab_pre_eeg = ap_fit_label(psd_pre, "Pre", **fooof_inp)
ap_seiz_eeg, lab_seiz_eeg = ap_fit_label(psd_seiz, "Seizure", **fooof_inp)
ap_post_eeg, lab_post_eeg = ap_fit_label(psd_post, "Post", **fooof_inp)
ap_pre_sim, lab_pre_sim = ap_fit_label(saw_pre, "Pre", **fooof_inp)
ap_seiz_sim, lab_seiz_sim = ap_fit_label(saw_seiz, "Seizure", **fooof_inp)
ap_post_sim, lab_post_sim = ap_fit_label(saw_post, "Post", **fooof_inp)

ticks_time = dict(length=6, width=1.5)
ticks_psd = dict(length=4, width=1)
# abc = dict(x=0, y=1.01, fontsize=14, fontdict=dict(fontweight="bold"))

n = 10  # step size plotting
xticks1 = np.arange(0, (post_seiz-pre_seiz+1)/n, step=(5*srate_ep/n))
# xticklabels = (xticks - pre/n) / (srate_ep/n)
# xticklabels = [(str(int(element))) for element in xticklabels]

# %% Plot

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams['font.size'] = 14


fig, axes = plt.subplots(2, 2,  figsize=[12, 6],
                         gridspec_kw=dict(width_ratios=[1, .6]))

ax = axes[0, 0]

ax.plot(seiz_data[::n], c=c_empirical, lw=1)

ax.set_xticks(xticks1)
ax.set_xticklabels([])
ax.set_xlim([-0.5/n, (post_seiz-pre_seiz)/n])
yticks = [-200, 0, 200]
ax.set_yticks(yticks)
ax.set_yticklabels([-200, "", 200])

rectangle_pre = plt.Rectangle((0, -1000),
                              (seiz_start-pre_seiz)/n,
                              9000,
                              alpha=0.2, color=c_pre)
ax.add_patch(rectangle_pre)

rectangle_seiz = plt.Rectangle(((seiz_start-pre_seiz)/n, -1000),
                               (seiz_end - seiz_start)/n,
                               9000,
                               alpha=0.2, color=c_seiz)
ax.add_patch(rectangle_seiz)

rectangle_post = plt.Rectangle(((seiz_end-pre_seiz)/n, -1000),
                               (post_seiz-seiz_end)/n,
                               9000,
                               alpha=0.2, color=c_post)
ax.add_patch(rectangle_post)
ax.set_ylabel(fr"{cha_nm} [$\mu$V]", labelpad=-20)
ax.tick_params(**ticks_time)
# ax.text(s="a", **abc, transform=ax.transAxes)


ax = axes[1, 0]

ax.plot(t[:30*srate_ep:n], noise_saw[::n], c=c_sim)
ylim = ax.get_ylim()

rectangle_pre = plt.Rectangle((0, ylim[0]),
                              10,
                              np.diff(ylim)[0],
                              alpha=0.2, color=c_pre)
ax.add_patch(rectangle_pre)

rectangle_seiz = plt.Rectangle((10, ylim[0]),
                               seiz_len/srate_ep,
                               np.diff(ylim)[0],
                               alpha=0.2, color=c_seiz)
ax.add_patch(rectangle_seiz)

rectangle_post = plt.Rectangle((10 + seiz_len/srate_ep, ylim[0]),
                               20 - seiz_len/srate_ep,
                               np.diff(ylim)[0],
                               alpha=0.2, color=c_post)
ax.add_patch(rectangle_post)

ax.set_xlim([0, 30])
xticks = np.arange(0, 35, 5)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks-10)
ax.set_xlabel("Time [s]")
yticks = [-3, 0, 3]
ax.set_yticks(yticks)
ax.set_yticklabels([])
ax.set_ylabel("Simulation [a.u.]", labelpad=15)
ax.tick_params(**ticks_time)


ax = axes[0, 1]

ax.loglog(freq, psd_pre, c_pre, lw=2)
ax.loglog(freq, psd_seiz, c_seiz, lw=2)
ax.loglog(freq, psd_post, c_post, lw=2)

ax.loglog(freq, ap_pre_eeg, "--", c=c_pre, lw=2, label=lab_pre_eeg)
ax.loglog(freq, ap_seiz_eeg, "--", c=c_seiz, lw=2, label=lab_seiz_eeg)
ax.loglog(freq, ap_post_eeg, "--", c=c_post, lw=2, label=lab_post_eeg)

ax.set_xlim(freq_range)

ax.legend(loc=1, fontsize=10)
ax.set_ylabel(r"PSD [$\mu$$V^2$/Hz]", fontsize=14, labelpad=-25)
ax.set_xlabel("")
xticks = [1, 10, 100]
ax.set_xticks(xticks)
ax.set_xticklabels([])
yticks = [1e-2, 1, 1e2, 1e4]
ax.set_yticks(yticks)
ax.set_yticklabels([r"$10^{-2}$", "", "", r"$10^4$"], fontsize=14)
ax.tick_params(**ticks_psd)
# ax.text(s="b", **abc, transform=ax.transAxes)

ax = axes[1, 1]

ax.loglog(freq, saw_pre, c_pre, lw=2)
ax.loglog(freq, saw_seiz, c_seiz, lw=2)
ax.loglog(freq, saw_post, c_post, lw=2)

ax.loglog(freq, ap_pre_sim, "--", c=c_pre, lw=2, label=lab_pre_sim)
ax.loglog(freq, ap_seiz_sim, "--", c=c_seiz, lw=2, label=lab_seiz_sim)
ax.loglog(freq, ap_post_sim, "--", c=c_post, lw=2, label=lab_post_sim)

ax.set_xlim(freq_range)

ax.legend(loc=1, fontsize=10)
ax.set_ylabel("")
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, fontsize=14)
yticks = [1e-3, 1e-1, 1e1, 1e3]
ax.set_yticks(yticks)
ax.set_yticklabels([])
ax.set_xlabel("Frequency [Hz]", fontsize=14)
ax.set_ylabel("PSD [a.u.]", fontsize=14, labelpad=10)
ax.tick_params(**ticks_psd)

plt.tight_layout()
plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()


# %% Plot Supp. b)

# best fooof params for seizure:
fooof_seiz_params = dict(peak_width_limits=(0, 1), peak_threshold=0)

# Message:
# even with these tuned params the fit is above 2 and therefore bad
# furthermore, one cannot tune fooof for each condition and then compare
# the 1/f

fm_seiz = FOOOF()
fm_saw = FOOOF()
fm_tuned_seiz = FOOOF(**fooof_seiz_params)
fm_tuned_saw = FOOOF(**fooof_seiz_params)

freq_range = [1, 100]

fm_seiz.fit(freq, psd_seiz, freq_range)
fm_saw.fit(freq, saw_seiz, freq_range)
fm_tuned_seiz.fit(freq, psd_seiz, freq_range)
fm_tuned_saw.fit(freq, saw_seiz, freq_range)

# %% Plot Supp
fig, axes = plt.subplots(2, 2, figsize=[12, 12], sharey="col")
ax = axes[0, 0]

fm_seiz.plot(ax=ax, plt_log=True)
exp_seiz = fm_seiz.get_params("aperiodic", "exponent")
ax.set_title(f"Seizure a={exp_seiz:.2f}")
ax.set_ylabel("Fooof default parameters")
ax.grid(False)

ax = axes[0, 1]

fm_saw.fit(freq, saw_seiz, freq_range)
fm_saw.plot(ax=ax, plt_log=True)
exp_seiz = fm_saw.get_params("aperiodic", "exponent")
ax.set_title(f"1/f={exp_seiz:.2f}")
ax.grid(False)
ax.set_ylabel("")

ax = axes[1, 0]

fm_tuned_seiz.plot(ax=ax, plt_log=True)
exp_seiz = fm_tuned_seiz.get_params("aperiodic", "exponent")
ax.set_title(f"1/f={exp_seiz:.2f}")
ax.set_ylabel("Fooof tuned parameters")
ax.grid(False)

ax = axes[1, 1]

fm_tuned_saw.plot(ax=ax, plt_log=True)
exp_seiz = fm_tuned_saw.get_params("aperiodic", "exponent")
ax.set_title(f"1/f={exp_seiz:.2f}")
ax.set_ylabel("")
ax.grid(False)

plt.suptitle(f"Fooof fit {freq_range[0]}-{freq_range[1]}Hz  sawtooth signal")
plt.savefig(fig_path + fig_name_supp, bbox_inches="tight")
plt.tight_layout()
plt.show()
