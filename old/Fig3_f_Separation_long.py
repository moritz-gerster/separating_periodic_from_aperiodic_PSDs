"""

Fooof cant distinguish peaks which are not clearly separable or ideally
Gaussian.

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
use epilepsy plot but make nice
"""
import numpy as np
import scipy.signal as sig
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
from fooof import FOOOF
from scipy.stats import norm


def osc_signals(samples, slopes, freq_osc=[], amp=[], width=[],
                srate=2400):
    """Simplified sim function."""
    # Initialize output
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


def calc_PSDs(channel_data, n_seiz, n_el, **welch_params):
    """Calc PSDs."""
    # n_freq = srate // 2 + 1
    nperseg = welch_params["nperseg"]
    n_freq = np.fft.rfftfreq(nperseg, d=1/srate).size

    shape = n_seiz, n_el, n_freq

    psds_pre = np.zeros(shape)
    psds_seiz = np.zeros(shape)
    psds_post = np.zeros(shape)

    for seiz in range(5):
        slice_full = slice(pre_tms[seiz][0], post_tms[seiz][1])
        seiz_len = post_tms[seiz][0] - pre_tms[seiz][1]

        # normalize
        data_full = channel_data[:, slice_full]
        data_full = data_full / data_full.std()

        data_pre = data_full[:, :10*srate]
        data_seiz = data_full[:, 10*srate:10*srate + seiz_len]
        data_post = data_full[:, 10*srate + seiz_len:30*srate]

        freq, psd_pre = sig.welch(data_pre, **welch_params)
        freq, psd_seiz = sig.welch(data_seiz, **welch_params)
        freq, psd_post = sig.welch(data_post, **welch_params)

        psds_pre[seiz] = psd_pre
        psds_seiz[seiz] = psd_seiz
        psds_post[seiz] = psd_post
    return psds_pre, psds_seiz, psds_post


def saw_noise(srate, pre_tms, post_tms, seiz, slope=[2], saw_power=0.9,
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
    noises, _ = osc_signals(samples, slope)
    noises = noises[0]

    # Highpass 0.3 Hz like real signal
    # sos = sig.butter(20, 1, btype="hp", fs=srate, output='sos')
    # noises = sig.sosfilt(sos, noises)

    # make signal 10 seconds zero, 10 seconds strong, 10 seconds zero
    pre = 10 * srate
    seiz_len = post_tms[seiz][0] - pre_tms[seiz][1]
    post = 20 * srate - seiz_len
    saw_seiz = np.r_[np.zeros(pre),
                     saw[:seiz_len],
                     np.zeros(post)]

    noise_saw = noises + saw_seiz

    # normalize
    noise_saw = (noise_saw - noise_saw.mean()) / noise_saw.std()
    return t, noise_saw


# %% Parameters c)

# Colors
c_empirical = "purple"
c_sim = "k"

c_seiz = "r"
c_pre = "c"
c_post = "y"

# Data Info c)
srate = 256
pre = 10 * srate
seiz_tms = [(88100, 91150), (167700, 169500), (193100, 195250),
            (441900, 444750), (472450, 475800)]

pre_tms = [(time[0] - pre, time[0]) for time in seiz_tms]
post_tms = [(time[1], time[1] + pre) for time in seiz_tms]

channel_nms = ['C4-P4', 'C3-P3', 'C3-Cz', 'C4-T4', 'Cz-C4',
               'F3-C3', 'F4-C4',
               'F7-T3', 'F8-T4',
               'Fp1-F7', 'Fp2-F8', 'Fp2-F4', 'Fp1-F3',
               'P3-O1', 'P4-O2',
               'T1-T3',  # 'T2-T1 delete',
               'T3-C3', 'T4-T2', 'T4-T6',
               'T6-O2', 'T3-T5', 'T5-O1']

ch_mat_nms = ['C1', 'C2', 'C3', 'C4', 'Cz',
              'F3', 'F4',
              'F7', 'F8',
              'Fp1', 'Fp2', 'Fp3', 'Fp4',
              'P3', 'P4',
              'T1',  # 'T2',
              'T3', 'T4', 'T5',
              'T6', 'T7', 'T8']

n_seiz = len(pre_tms)
n_el = len(channel_nms)

nperseg = srate
saw_power = 0.004
saw_width = 0.69
seed = 2
slope=[1.8]
ch = 5
seiz = 0
welch_params = {"fs": srate, "nperseg": nperseg}

# Paths
fig_path = "../paper_figures/"
fig_name = "Fig3_f_Separation.pdf"
fig_name_supp = "Fig3_f_Separation_SuppMat.pdf"
data_path = "../../1-f_Absence_Seizure/data/"
# %% Load data

channel_data = np.squeeze([sio.loadmat(data_path + f"{name}.mat")[name]
                           for name in ch_mat_nms])

psds_pre, psds_seiz, psds_post = calc_PSDs(channel_data, n_seiz, n_el,
                                           **welch_params)

# %% Make signal

t, noise_saw = saw_noise(srate, pre_tms, post_tms, seiz, slope=slope,
                         saw_power=saw_power,
                         saw_width=saw_width, seed=seed)

seiz_len = post_tms[seiz][0] - pre_tms[seiz][1]

freq, saw_pre = sig.welch(noise_saw[:pre], **welch_params)
freq, saw_seiz = sig.welch(noise_saw[pre:seiz_len+pre], **welch_params)
freq, saw_post = sig.welch(noise_saw[seiz_len + pre:], **welch_params)


# %% Plot

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
start = pre_tms[seiz][0]
end = start + 30*srate
slice_full = slice(start, end)

data_full = channel_data[:, slice_full]
# normalize
data_full = data_full / data_full.std()

ax.plot(data_full[ch, ::n], c=c_empirical, lw=1)

xticks = np.arange(0, (end-start+1)/n, step=(5*srate/n))
xticklabels = (xticks - pre/n) / (srate/n)
xticklabels = [(str(int(element))) for element in xticklabels]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim([-0.5/n, (end-start)/n])

rectangle_pre = plt.Rectangle((0, -1000),
                              (pre_tms[seiz][1]-pre_tms[seiz][0])/n,
                              9000,
                              alpha=0.2, color=c_pre)
ax.add_patch(rectangle_pre)

rectangle_seiz = plt.Rectangle(((pre_tms[seiz][1]-pre_tms[seiz][0])/n, -1000),
                               (post_tms[seiz][0] - pre_tms[seiz][1])/n,
                               9000,
                               alpha=0.2, color=c_seiz)
ax.add_patch(rectangle_seiz)

rectangle_post = plt.Rectangle(((post_tms[seiz][0]-pre_tms[seiz][0])/n, -1000),
                               (np.diff(post_tms[seiz])[0])/n,
                               9000,
                               alpha=0.2, color=c_post)
ax.add_patch(rectangle_post)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel(f"{channel_nms[ch]} 30s SD")


ax = axes[1, 0]

ax.plot(t[:30*srate:n], noise_saw[::n], c=c_sim)
ylim = ax.get_ylim()

rectangle_pre = plt.Rectangle((0, ylim[0]),
                              10,
                              np.diff(ylim)[0],
                              alpha=0.2, color=c_pre)
ax.add_patch(rectangle_pre)

rectangle_seiz = plt.Rectangle((10, ylim[0]),
                               seiz_len/srate,
                               np.diff(ylim)[0],
                               alpha=0.2, color=c_seiz)
ax.add_patch(rectangle_seiz)

rectangle_post = plt.Rectangle((10 + seiz_len/srate, ylim[0]),
                               20 - seiz_len/srate,
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

fm = FOOOF()
labels = []

fm.fit(freq, psds_pre[seiz, ch], freq_range)
fm.plot(ax=ax, **pre_kwargs)
exp_pre = fm.get_params("aperiodic", "exponent")
labels.append(f"1/f Pre={exp_pre:.2f}")

fm.fit(freq, psds_seiz[seiz, ch], freq_range)
fm.plot(ax=ax, **seiz_kwargs)
exp_seiz = fm.get_params("aperiodic", "exponent")
labels.append(f"1/f Seizure={exp_seiz:.2f}")

fm.fit(freq, psds_post[seiz, ch], freq_range)
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



diff = exp_seiz - np.mean([exp_pre, exp_post])
axes[0, 0].set_title(f"1/f diff = {diff:.2f}, seiz={seiz}")


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

diff = exp_seiz - np.mean([exp_pre, exp_post])
axes[1, 0].set_title(f"1/f diff = {diff:.2f}, saw_width={saw_width}")

plt.tight_layout()
#plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()


# %% Supp. MAt.


fm = FOOOF(peak_width_limits=(0, 12), peak_threshold=2)

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
#plt.savefig(fig_path + fig_name_supp, bbox_inches="tight")
plt.show()


"""
Make script shorter

draft b)

draft a)
"""