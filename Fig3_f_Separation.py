"""
Fooof needs clearly separable (and ideally Gaussian) peaks.

a)
1) show pure 1/f
2) show clearly separable Gaussians oscillations
3) show sum and successful fooof fit

2) show strongly overlappy oscillations by adding intermediate oscillations
    in the middle (dashed)
3) fooof underestimates 1/f because 1/f is hidden below oscillations

b)
1) show easy spectrum with fooof fit and 2 other possibilites
2) show "hard" spectrum with fooof fit and 2 other possibilites

c)
use epilepsy plot but make nice - CHECK
"""
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
from fooof import FOOOF
from scipy.stats import norm
import mne
from mne.time_frequency import psd_welch
from fooof.sim.gen import gen_aperiodic


# %% Functions c)


def osc_signals(samples, slopes, freq_osc=[], amp=[], width=[],
                srate=2400):
    """Simplified sim function."""
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
              saw_width=0.54, freq=3, duration=30, seed=1):
    """Make Docstring."""
    if seed:
        np.random.seed(seed)
    # duration and sampling
    srate = srate
    time = np.arange(duration * srate)
    samples = time.size

    # Sawtooth signal
    t = np.arange(0, duration, 1/srate)
    saw = sawtooth(2 * np.pi * freq * t, width=saw_width)
    saw = saw[:-2]
    saw *= saw_power  # scaling

    # 1/f noise
    noises, _ = osc_signals(samples, slope, srate=srate_ep)
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


# %% Parameters b)

srate_pd = 2400
# %% Parameters c)

# Colors
c_empirical = "purple"
c_sim = "k"

c_seiz = "r"
c_pre = "c"
c_post = "y"

srate_ep = 256
seiz_start = 88100
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
welch_params = {"fs": srate_ep, "nperseg": nperseg}
saw_params = dict(slope=slope, saw_power=saw_power, saw_width=saw_width,
                  seed=seed)
fooof_params = dict(verbose=False)  # standard params

# Paths
data_path = "../data/Fig3/"
fig_path = "../paper_figures/"
fig_name = "Fig3_f_Separation.pdf"
fig_name_supp = "Fig3_f_Separation_SuppMat.pdf"

# %% Get data b)

sub5 = mne.io.read_raw_fif(data_path + "subj5_on_R1_raw.fif", preload=True)
sub9 = mne.io.read_raw_fif(data_path + "subj9_on_R8_raw.fif", preload=True)

ch5 = "SMA"
ch9 = "STN_R01"
sub5.pick_channels([ch5])
sub9.pick_channels([ch9])
filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": 0.1,
                 "method": "spectrum_fit"}
sub5.notch_filter(**filter_params)
sub9.notch_filter(**filter_params)

welch_params = {"fmin": 1,
                "fmax": 600,
                "tmin": 0.5,
                "tmax": 185,
                "n_fft": srate_pd,
                "n_overlap": srate_pd // 2,
                "average": "mean"}

spec5, freq = psd_welch(sub5, **welch_params)
spec9, freq = psd_welch(sub9, **welch_params)

spec5 = spec5[0]
spec9 = spec9[0]

# %% Plot b)
fm5 = FOOOF(**fooof_params)
fm9 = FOOOF(**fooof_params)

freq_range = [1, 95]
fm5.fit(freq, spec5, freq_range)
fm9.fit(freq, spec9, freq_range)

fm5_ap = gen_aperiodic(fm5.freqs, fm5.aperiodic_params_)
fm9_ap = gen_aperiodic(fm9.freqs, fm9.aperiodic_params_)

fig, ax = plt.subplots(2, 2)
ax[0, 0].semilogy(freq, spec5, c_empirical, label="Sub 5 " + ch5)
ax[0, 1].semilogy(freq, spec9, c_empirical, label="Sub 9 " + ch9)
ax[1, 0].loglog(freq, spec5, c_empirical)
ax[1, 1].loglog(freq, spec9, c_empirical)

# Fooof fit
ax[0, 0].semilogy(fm5.freqs, 10**fm5_ap, "b--",
                  label=f"fooof a={fm5.aperiodic_params_[1]:.2f}")
ax[0, 1].semilogy(fm9.freqs, 10**fm9_ap, "b--",
                  label=f"fooof a={fm9.aperiodic_params_[1]:.2f}")
ax[1, 0].loglog(fm5.freqs, 10**fm5_ap, "b--")
ax[1, 1].loglog(fm9.freqs, 10**fm9_ap, "b--")

# Straight fit
DeltaX = np.diff(freq_range)[0]
DeltaY5 = spec5[freq==freq_range[0]][0] - spec5[freq==freq_range[1]][0]
a5 = DeltaX / DeltaY5
ax[0, 0].semilogy(freq_range, [spec5[freq==freq_range[0]][0], spec5[freq==freq_range[1]][0]],
                  "k+",
                  label=f"fooof a={fm5.aperiodic_params_[1]:.2f}")
ax[1, 0].loglog(freq_range, [spec5[freq==freq_range[0]][0], spec5[freq==freq_range[1]][0]],
                  "k+",
                  label=f"fooof a={fm5.aperiodic_params_[1]:.2f}")

# Low fit

for axes in ax.flatten():
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
ax[0, 0].set(xlabel=None, ylabel=r"PSD [$\mu$$V^2/Hz$]")
ax[0, 1].set(xlabel=None, ylabel=None)
ax[1, 0].set(xlabel="Frequency [Hz]", ylabel=r"PSD [$\mu$$V^2/Hz$]")
ax[1, 1].set(xlabel="Frequency [Hz]", ylabel=None)
ax[0, 0].legend()
ax[0, 1].legend()
plt.tight_layout()
plt.show()
# %% Get data c)

seiz_data = np.load(data_path + cha_nm + ".npy", allow_pickle=True)

# normalize
seiz_data = seiz_data[pre_seiz:post_seiz]
seiz_data /= seiz_data.std()


t, noise_saw, freq, saw_pre, saw_seiz, saw_post = saw_noise(srate_ep, seiz_len,
                                                            **saw_params)

freq, psd_pre, psd_seiz, psd_post = calc_PSDs(seiz_data, seiz_len, srate_ep,
                                              **welch_params)


# %% Plot c)

pre_kwargs = dict(plt_log=True,
                  model_kwargs={"alpha": 0},
                  data_kwargs={"alpha": 1, "color": c_pre},
                  aperiodic_kwargs={"color": c_pre, "alpha": 1})
seiz_kwargs = dict(plt_log=True,
                   model_kwargs={"alpha": 0},
                   data_kwargs={"alpha": 1, "color": c_seiz},
                   aperiodic_kwargs={"color": c_seiz, "alpha": 1})
post_kwargs = dict(plt_log=True,
                   model_kwargs={"alpha": 0},
                   data_kwargs={"alpha": 1, "color": c_post},
                   aperiodic_kwargs={"color": c_post, "alpha": 1})


fig, axes = plt.subplots(2, 2,  figsize=[12, 6],
                         gridspec_kw=dict(width_ratios=[1, .6]))

ax = axes[0, 0]

n = 10  # step size plotting

ax.plot(seiz_data[::n], c=c_empirical, lw=1)

xticks = np.arange(0, (post_seiz-pre_seiz+1)/n, step=(5*srate_ep/n))
xticklabels = (xticks - pre/n) / (srate_ep/n)
xticklabels = [(str(int(element))) for element in xticklabels]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim([-0.5/n, (post_seiz-pre_seiz)/n])

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

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel(f"{cha_nm} 30s SD")


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
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Simulation")


ax = axes[0, 1]

freq_range = [1, 100]

fm = FOOOF(**fooof_params)
labels = []

fm.fit(freq, psd_pre, freq_range)
fm.plot(ax=ax, **pre_kwargs)
exp_pre = fm.get_params("aperiodic", "exponent")
labels.append(f"1/f Pre={exp_pre:.2f}")

fm.fit(freq, psd_seiz, freq_range)
fm.plot(ax=ax, **seiz_kwargs)
exp_seiz = fm.get_params("aperiodic", "exponent")
labels.append(f"1/f Seizure={exp_seiz:.2f}")

fm.fit(freq, psd_post, freq_range)
fm.plot(ax=ax, **post_kwargs)
exp_post = fm.get_params("aperiodic", "exponent")
labels.append(f"1/f Post={exp_post:.2f}")

handles, _ = ax.get_legend_handles_labels()
handles = handles[2::3]
ax.legend(handles, labels, loc=1)
ax.grid(False)
ax.set_ylabel("PSD")
ax.set_xlabel("")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
xticks = [1, 10, 100]
xticks_pos = np.log10(np.array(xticks))
ax.set_xticks(xticks_pos)
ax.set_xticklabels([])

ax = axes[1, 1]

labels = []

fm.fit(freq, saw_pre, freq_range)
fm.plot(ax=ax, **pre_kwargs)
exp_pre = fm.get_params("aperiodic", "exponent")
labels.append(f"1/f Pre={exp_pre:.2f}")

fm.fit(freq, saw_seiz, freq_range)
fm.plot(ax=ax, **seiz_kwargs)
exp_seiz = fm.get_params("aperiodic", "exponent")
labels.append(f"1/f Seizure={exp_seiz:.2f}")

fm.fit(freq, saw_post, freq_range)
fm.plot(ax=ax, **post_kwargs)
exp_post = fm.get_params("aperiodic", "exponent")
labels.append(f"1/f Post={exp_post:.2f}")

ax.legend(handles, labels, loc=1)
ax.grid(False)
ax.set_ylabel("PSD")
ax.set_xticks(xticks_pos)
ax.set_xticklabels(xticks)
ax.set_xlabel("Frequency [Hz]")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()


# %% Plot Supp. c)


fm = FOOOF(**fooof_params)

freq_range = [1, 100]

fig, axes = plt.subplots(1, 2, figsize=[12, 6], sharey=True)
ax = axes[0]

fm.fit(freq, saw_seiz, freq_range)
fm.plot(ax=ax)
exp_seiz = fm.get_params("aperiodic", "exponent")
ax.set_title(f"1/f={exp_seiz:.2f}")
ax.grid(False)

ax = axes[1]
fm.plot(ax=ax, plt_log=True)
ax.grid(False)
ax.set_title(f"1/f={exp_seiz:.2f}")
plt.suptitle(f"Fooof fit {freq_range[0]}-{freq_range[1]}Hz  sawtooth signal")
plt.savefig(fig_path + fig_name_supp, bbox_inches="tight")
plt.show()
