"""
Simulate 1/f power spectra to compare fooof 1/f-estimation vs. IRASA.

1. Create 1/f noise for different exponents.
2. Fit fooof and IRASA. Test how well they obtain the 1/f-exponents.
4. Vary parameters such as nperseg in Welch
3. Add Gaussian white noise. Test how the noise affects the 1/f estimate
   for different frequency ranges.
5. Add one very strong sine peak. Test if 1/f is still fitted well.
   Test if fooof and IRASA detect the peak power and frequency and width
   correctly.
6. Vary the amplitude (SNR) and width of the peak.
7. Vary fitting ranges.
8. Add more peaks. Some strong, some weak, some wide, small sharp, some
   overlapping, some distinct. Which algorithm performs best?
9. Add noise to peaks
"""
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from noise_helper import noise_white, osc_signals, psds_pink, slope_error
from noise_helper import plot_all, irasa

params = {'legend.fontsize': 12.2,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'xtick.minor.size': 0,
          'xtick.major.size': 5,
          'ytick.major.size': 5,
          'lines.linewidth': 2}
plt.rcParams.update(params)



# %% PARAMETERS


# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slopes = np.arange(0, 4.5, .5)

# WELCH
win_sec = 4
nperseg = int(win_sec * srate)
nperseg_fooof = int(1 * srate)  # 4*srate too high resolution for fooof

# Fit Params
# fooof_params = {"peak_width_limits": (4, 12)}
irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                "kwargs_welch": {'average': 'mean'}}

# Path
fig_path = "../plots/"
white_ratio = 0


# %% White noise variation

folder = "white_noise"
save_path = fig_path + f"{folder}/"
freq_range = [2, 100]

white_ratios = [0, 0.001, 0.05, 1]

# No oscillations
freq_osc, amp = None, None

# Make noise
noises = osc_signals(samples, slopes, freq_osc, amp)
w_noise = noise_white([slopes.size, samples-2])

# Initialize
errs_f = []
errs_i = []
for white_ratio in white_ratios:
    noise_mix = noises + white_ratio * w_noise
    freq, noise_psds = psds_pink(noise_mix, srate, nperseg)
    freq_f, noise_psds_f = psds_pink(noise_mix, srate, nperseg_fooof)

    save_name = f"noise={white_ratio}.pdf"
    IRASA = irasa(data=noise_mix, band=freq_range, **irasa_params)

    fooof_params = dict(max_n_peaks=0, verbose=False)  # no oscillations
    err_f, err_i = slope_error(slopes, freq_f, noise_psds_f, freq_range, IRASA,
                               fooof_params=fooof_params)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    plot_all(freq, noise_psds, freq_f, noise_psds_f, IRASA, slopes,
             freq_range, white_ratio, save_path=save_path, save_name=save_name)

data = {"freq_ranges": [freq_range] * len(white_ratios),
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(white_ratios),
        "white_ratio": white_ratios}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_noise_vary.pkl"
df.to_pickle(save_path + save_name)


fig, ax = plt.subplots(1, 1)
ax.plot(df["white_ratio"], df["err_f"], "r--", label="fooof")
ax.plot(df["white_ratio"], df["err_i"], "b:", label="IRASA")
ax.set_xlabel("White Noise ratio")
ax.set_ylabel("Fitting error")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
plt.title("Realistic fitting ranges")
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()


# %% Welch variation

# very slow


folder = "welch_window"
save_path = fig_path + f"{folder}/"
freq_range = [1, 100]
welch_windows = [0.25, 0.5, 1, 2, 4]

# No oscillations
freq_osc, amp = None, None

# Make noise
noises = osc_signals(samples, slopes, freq_osc, amp)
white_ratio = 0

# Initilaiize
errs_f = []
errs_i = []
for win_sec in welch_windows:
    nperseg = int(win_sec * srate)
    freq, noise_psds = psds_pink(noises, srate, nperseg)
    irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                    "kwargs_welch": {'average': 'mean'}}

    save_name = f"welch_window={win_sec}.pdf"
    IRASA = irasa(data=noises, band=freq_range, **irasa_params)

    fooof_params = dict(max_n_peaks=0, verbose=False)  # no oscillations
    err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA,
                               fooof_params=fooof_params)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    plot_all(freq, noise_psds, freq, noise_psds, IRASA, slopes, freq_range,
             white_ratio, save_path=save_path, save_name=save_name)

data = {"freq_ranges": [freq_range] * len(welch_windows),
        "welch_window": welch_windows,
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(welch_windows),
        "noise_gauss": [white_ratio] * len(welch_windows)}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_welch_window_vary.pkl"
df.to_pickle(save_path + save_name)


fig, ax = plt.subplots(1, 1)
ax.plot(df["welch_window"], df["err_f"], "r--",
        alpha=1, label="fooof")
ax.plot(df["welch_window"], df["err_i"], "b:",
        alpha=1, label="IRASA")
ax.set_xlabel("Welch Window [s]")
ax.set_ylabel("Fitting error")
ax.legend()
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()


# %% IRASA not realiable for high frequencies

folder = "IRASA_high_freq"
save_path = fig_path + f"{folder}/"
freq_ranges = [[10, 110], [210, 310], [310, 410], [360, 460],
               [410, 510], [460, 560], [510, 610]]

# Make noise
noises = osc_signals(samples, slopes, None, None)
white_ratio = 0

# Initilaiize
errs_f = []
errs_i = []
for freq_range in freq_ranges:
    freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
    save_name = f"{freq_name}.pdf"
    freq_f, noise_psds_f = psds_pink(noises, srate, nperseg_fooof)
    IRASA = irasa(data=noises, band=freq_range, **irasa_params)

    fooof_params = dict(max_n_peaks=0, verbose=False)  # no oscillations
    err_f, err_i = slope_error(slopes, freq_f, noise_psds_f, freq_range, IRASA,
                               fooof_params=fooof_params)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    plot_all(freq, noise_psds, freq_f, noise_psds_f, IRASA, slopes, freq_range,
             white_ratio, save_path=save_path, save_name=save_name)


data = {"freq_ranges": freq_ranges,
        "welch_window": [win_sec] * len(freq_ranges),
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(freq_ranges),
        "noise_gauss": [white_ratio] * len(freq_ranges)}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_high_freq_vary.pkl"
df.to_pickle(save_path + save_name)


fig, ax = plt.subplots(1, 1)
for i in range(df.shape[0]):
    ax.plot(df["freq_ranges"][i], [df["err_f"][i], df["err_f"][i]], "r--",
            alpha=1, label="fooof")
    ax.plot(df["freq_ranges"][i], [df["err_i"][i], df["err_i"][i]], "b:",
            alpha=1, label="IRASA")
ax.set_xlabel("Frequency Range [Hz]")
ax.set_ylabel("Fitting error")
ymin, ymax = ax.get_ylim()
ax.set_ylim([ymin, ymax])
ax.vlines((srate/1.9)/2, ymin, ymax, color="k", label="Resample Nyquist Freq.")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2] + [handles[-1]], labels[:2] + [labels[-1]])
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()


# %% Add sine peaks (plot wrong)


folder = "pure_sine_peaks"
save_path = fig_path + f"{folder}/"

freq_range = [30, 50]

# Generate signal
freq = 40  # Hz
amp = 100
signals = osc_signals(samples, slopes, freq, amp)

freq, noise_psds = psds_pink(signals, srate, nperseg)
freq_f, noise_psds_f = psds_pink(signals, srate, nperseg_fooof)
freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
save_name = f"{freq_name}.pdf"
# IRASA = irasa(data=signals, band=freq_range, **irasa_params)

plot_all(freq, noise_psds, freq_f, noise_psds_f, IRASA, slopes, freq_range,
         white_ratio, plot_osc=True, save_path=save_path, save_name=save_name)


# %% IRASA neees larger up/down sampling for more broad widths


folder = "oscillation_widths"
save_path = fig_path + f"{folder}/"
freq_range = [5, 40]
white_ratio = 0
win_sec = 4


# Oscillation
freq_osc = [10, 25]  # Hz
amp = [2, 10]

# Initilaiize
df = pd.DataFrame()
hset_maxis = [1.9, srate / 4 / freq_range[1]]
for hset_max in hset_maxis:
    irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                    "kwargs_welch": {'average': 'mean'},
                    "hset": np.linspace(1.1, hset_max, num=18)}
    beta_widths = [0.1, 2, 4, 6, 8]
    errs_f = []
    errs_i = []
    for beta_width in beta_widths:
        width = [0.5, beta_width]
        signals = osc_signals(samples, slopes, freq_osc, amp, width=width)
        freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
        save_name = (f"{freq_name}_beta_width={beta_width}_"
                     f"hset_max={hset_max:.2f}.pdf")
        add_title = f"beta width={beta_width} SDs, hmax={hset_max:.2f}"
        freq, noise_psds = psds_pink(signals, srate, nperseg)

        freq_f, noise_psds_f = psds_pink(signals, srate, nperseg_fooof)
        IRASA = irasa(data=signals, band=freq_range, **irasa_params)

        fooof_params = dict(max_n_peaks=0, verbose=False)  # no oscillations
        err_f, err_i = slope_error(slopes, freq_f, noise_psds_f, freq_range,
                                   IRASA, fooof_params=fooof_params)
        errs_f.append(np.sum(np.abs(err_f)))
        errs_i.append(np.sum(np.abs(err_i)))

        # plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
          #     save_path=save_path, save_name=save_name, add_title=add_title)
        plot_all(freq, noise_psds, freq_f, noise_psds_f, IRASA, slopes,
                 freq_range, white_ratio,
                 plot_osc=True,
                 save_path=save_path, save_name=save_name, add_title=add_title)

    data = {"freq_ranges": [freq_range] * len(beta_widths),
            "welch_window": [win_sec] * len(beta_widths),
            "err_f": errs_f, "err_i": errs_i,
            "vary": [folder] * len(beta_widths),
            "amp": [amp] * len(beta_widths),
            "freq_osc": [freq_osc] * len(beta_widths),
            "width": beta_widths,
            "hset_max": [hset_max] * len(beta_widths),
            "noise_gauss": [white_ratio] * len(beta_widths)}
    temp = pd.DataFrame(data)
    df = df.append(temp)

save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_beta_width_hmax_vary.pkl"
df.to_pickle(save_path + save_name)

df1 = df[df.hset_max == 1.9]
df2 = df[df.hset_max == 15]

fig, ax = plt.subplots(1, 1)
ax.plot(df1["width"], df1["err_f"], "r--",
        alpha=1, label="fooof")
ax.plot(df1["width"], df1["err_i"], "b:",
        alpha=1, label="IRASA hmax=1.9")
ax.plot(df2["width"], df2["err_i"], "b-",
        alpha=1, label="IRASA hmax=15")
ax.set_xlabel("Gauss width SDs")
ax.set_ylabel("Fitting error")
ax.legend()
# plt.title(f"IRASA hmin={1/hset_max:.1f}")
plt.tight_layout()
plt.savefig(fig_path + folder + "/hset_max_summary.pdf", bbox_inches="tight")
plt.show()


# %% peak_extends_fitting_range

# plot wrong

folder = "peak_extends_fitting_range"
save_path = fig_path + f"{folder}/"
freq_ranges = [[1, 50], [2, 50], [3, 50], [4, 50], [5, 50], [6, 50], [7, 50],
               [10, 50], [12, 50], [20, 50]]

freq_range = [2, 50]
hset_max = srate / 4 / freq_range[1]
irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                "kwargs_welch": {'average': 'mean'},
                "hset": np.linspace(1.1, hset_max, num=18)}

# Oscillation
# freq_osc = [3, 6, 10, 18, 20]  # Hz
# amp = [4, 4, 4, 5, 10]
# width = [.5, .3, .5, 2, 3]
freq_osc = [3, 6, 10, 20]  # Hz
amp = [4, 4, 4, 7]
width = [.5, .3, .5, 3]

# Initilaiize
errs_f = []
errs_i = []
for freq_range in freq_ranges:
    freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
    save_name = f"{freq_name}.pdf"
    add_title = f"Peaks at {freq_osc}Hz"
    signals = osc_signals(samples, slopes, freq_osc, amp, width=width)
    freq, noise_psds = psds_pink(signals, srate, nperseg)
    freq_f, noise_psds_f = psds_pink(signals, srate, nperseg_fooof)
    IRASA = irasa(data=signals, band=freq_range, **irasa_params)

    err_f, err_i = slope_error(slopes, freq_f, noise_psds_f, freq_range, IRASA,
                               fooof_params=None)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    # plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
    # save_path=save_path, save_name=save_name, add_title=add_title)
    plot_all(freq, noise_psds, freq_f, noise_psds_f, IRASA, slopes, freq_range,
             white_ratio,
             plot_osc=True,
             save_path=save_path, save_name=save_name, add_title=add_title)

data = {"freq_ranges": freq_ranges,
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(freq_ranges),
        "win_sec": [win_sec] * len(freq_ranges),
        "resampling_max": [hset_max] * len(freq_ranges),
        "amp": [amp] * len(freq_ranges),
        "widths": [width] * len(freq_ranges),
        "freq_oscs": [freq_osc] * len(freq_ranges),
        "noise_gauss": [white_ratio] * len(freq_ranges)}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_fit_range_low.pkl"
df.to_pickle(save_path + save_name)

fig, ax = plt.subplots(1, 1)

df["f_low"] = np.array(freq_ranges)[:, 0]

ax.plot(df["f_low"], df["err_f"], "r--",
        alpha=1, label="fooof")
ax.plot(df["f_low"], df["err_i"], "b:",
        alpha=1, label="IRASA")
ax.set_xticks(freq_osc)
ax.set_xlim([0, 13])
ax.set_ylim([0, 13])
ax.set_xlabel("Lower Frequency Fitting Border [Hz]")
ax.set_ylabel("Fitting error")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
plt.title(add_title)
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()


# %% SNR

folder = "SNR"
save_path = fig_path + f"{folder}/"


freq_range = [5, 45]
hset_max = srate / 4 / freq_range[1]
irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                "kwargs_welch": {'average': 'mean'},
                "hset": np.linspace(1.1, hset_max, num=18)}

# Oscillation
freq_osc = [10, 18, 20]  # Hz
amp = np.array([2, 5, 10])
width = [.5, 2, 4]

# Initilaiize
errs_f = []
errs_i = []
amp_mods = [0.01, .1, .5]
for amp_mod in amp_mods:
    freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
    save_name = f"{freq_name}_amp-mod={amp_mod}.pdf"
    signals = osc_signals(samples, slopes, freq_osc, amp*amp_mod, width=width)
    freq, noise_psds = psds_pink(signals, srate, nperseg)
    freq_f, noise_psds_f = psds_pink(signals, srate, nperseg_fooof)
    IRASA = irasa(data=signals, band=freq_range, **irasa_params)

    fooof_params = dict(max_n_peaks=0, verbose=False)  # no oscillations
    err_f, err_i = slope_error(slopes, freq, noise_psds, freq_range, IRASA,
                               fooof_params=fooof_params)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    # plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
      #       save_path=save_path, save_name=save_name)
    plot_all(freq, noise_psds, freq_f, noise_psds_f, IRASA, slopes, freq_range,
             white_ratio,
             plot_osc=True,
             save_path=save_path, save_name=save_name)

data = {"freq_ranges": [freq_range] * len(amp_mods),
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(amp_mods),
        "win_sec": [win_sec] * len(amp_mods),
        "resampling_max": [hset_max] * len(amp_mods),
        "amp": [amp] * len(amp_mods),
        "amp_mods": amp_mods,
        "widths": [width] * len(amp_mods),
        "freq_oscs": [freq_osc] * len(amp_mods),
        "noise_gauss": [white_ratio] * len(amp_mods)}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_SNR.pkl"
df.to_pickle(save_path + save_name)

fig, ax = plt.subplots(1, 1)

ax.plot(df["amp_mods"], df["err_f"], "r--",
        alpha=1, label="fooof")
ax.plot(df["amp_mods"], df["err_i"], "b:",
        alpha=1, label="IRASA")
ax.set_xlabel("Amplitude modulation a.u.")
ax.set_ylabel("Fitting error")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
plt.title("Oscillatory peaks at 10, 18, and 20Hz")
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()


# %% realistic + noise


folder = "realistic+noise"
save_path = fig_path + f"{folder}/"
freq_range = [5, 45]

# Oscillation
freq_osc = [10, 18, 20]  # Hz
amp = [2, 5, 10]
width = [.5, 2, 4]
white_ratios = [0.001, .01, .05, .1, .2]

# Initilaiize
errs_f = []
errs_i = []
w_noise = noise_white([slopes.size, samples-2])

for white_ratio in white_ratios:
    freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
    save_name = f"{freq_name}_amp-mod={amp_mod}.pdf"
    signals = osc_signals(samples, slopes, freq_osc, amp, width=width)
    signals += white_ratio * w_noise
    freq, noise_psds = psds_pink(signals, srate, nperseg)
    freq_f, noise_psds_f = psds_pink(signals, srate, nperseg_fooof)
    IRASA = irasa(data=signals, band=freq_range, **irasa_params)

    fooof_params = dict(max_n_peaks=0, verbose=False)  # no oscillations
    err_f, err_i = slope_error(slopes, freq_f, noise_psds_f, freq_range, IRASA,
                               fooof_params=fooof_params)
    errs_f.append(np.sum(np.abs(err_f)))
    errs_i.append(np.sum(np.abs(err_i)))

    # plot_osc(freq, noise_psds, IRASA, slopes, freq_range, white_ratio,
    #       save_path=save_path, save_name=save_name)
    plot_all(freq, noise_psds, freq_f, noise_psds_f, IRASA, slopes, freq_range,
             white_ratio,
             plot_osc=True,
             save_path=save_path, save_name=save_name)

data = {"freq_ranges": [freq_range] * len(white_ratios),
        "err_f": errs_f, "err_i": errs_i,
        "vary": [folder] * len(white_ratios),
        "win_sec": win_sec * len(white_ratios),
        "resampling_max": [hset_max] * len(white_ratios),
        "amp": [amp] * len(white_ratios),
        "widths": [width] * len(white_ratios),
        "freq_oscs": [freq_osc] * len(white_ratios),
        "noise_gauss": white_ratios}
df = pd.DataFrame(data)
save_path = fig_path + folder + "/"
Path(save_path).mkdir(parents=True, exist_ok=True)
save_name = "df_noise.pkl"
df.to_pickle(save_path + save_name)


fig, ax = plt.subplots(1, 1)
ax.plot(df["noise_gauss"], df["err_f"], "r--",
        alpha=1, label="fooof")
ax.plot(df["noise_gauss"], df["err_i"], "b:",
        alpha=1, label="IRASA")
xticks = ax.get_xticks()
ax.set_xticklabels([int(xtick*100) for xtick in xticks])
ax.set_xlabel("White Noise [%]")
ax.set_ylabel("Fitting error")
ax.legend()
# plt.title("Realistic fitting ranges")
plt.tight_layout()
plt.savefig(fig_path + folder + "/summary.pdf", bbox_inches="tight")
plt.show()


# %% IRASA fucks up when highpass filtering


fig_path = "../plots/"

folder = "highpass"
save_path = fig_path + f"{folder}/"
freq_range = [5, 95]
white_ratio = 0
win_sec = 4


# Oscillation
freq_osc = [10, 25]  # Hz
amp = [2, 10]


hset_max = srate / 4 / freq_range[1]
hset_max = 9.9

irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                "kwargs_welch": {'average': 'mean'},
                "hset": np.linspace(1.1, hset_max, num=5)}
# beta_widths = [0.1, 2, 4]
beta_width = 2
width = [0.5, beta_width]
signals = osc_signals(samples, slopes, freq_osc, amp, width=width)

# highpass filter
sos = sig.butter(10, 1, btype="hp", fs=srate, output='sos')
signals = sig.sosfilt(sos, signals)
freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
save_name = (f"{freq_name}_beta_width={beta_width}_"
             f"hset_max={hset_max:.2f}.pdf")
add_title = f"beta width={beta_width} SDs, hmax={hset_max:.2f}"
freq, noise_psds = psds_pink(signals, srate, nperseg)
IRASA = irasa(data=signals, band=freq_range, **irasa_params)

plot_all(freq, noise_psds, freq, noise_psds, IRASA, slopes, freq_range,
         white_ratio, plot_osc=True,
         save_path=save_path, save_name=save_name, add_title=add_title)

# %% IRASA fucks up when noise floor


fig_path = "../plots/"

folder = "noise_floor"
save_path = fig_path + f"{folder}/"
freq_range = [15, 200]
white_ratio = .6
win_sec = 1
nperseg = int(win_sec * srate)

slopes = np.array([1, 1.6])
# Oscillation
freq_osc = 50  # Hz
amp = 1
w_noise = noise_white([slopes.size, samples-2])


# hset_max = srate / 4 / freq_range[1]
hset_max = 9.9

irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                "kwargs_welch": {'average': 'mean'},
                "hset": np.linspace(1.1, hset_max, num=5)}

width = 6
signals = osc_signals(samples, slopes, freq_osc, amp, width=width)
signals = signals + white_ratio * w_noise

# highpass filter
sos = sig.butter(10, 1, btype="hp", fs=srate, output='sos')
signals = sig.sosfilt(sos, signals)
freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
save_name = (f"{freq_name}_beta_width={width}_"
             f"hset_max={hset_max:.2f}.pdf")
add_title = f"beta width={width} SDs, hmax={hset_max:.2f}"
freq, noise_psds = psds_pink(signals, srate, nperseg)
IRASA = irasa(data=signals, band=freq_range, **irasa_params)

plot_all(freq, noise_psds, freq, noise_psds, IRASA, slopes, freq_range,
         white_ratio,
         plot_osc=True,
         save_path=save_path, save_name=save_name, add_title=add_title)


# %% IRASA fine when using intermediate ranges?

fig_path = "../plots/"

folder = "IRASA_works"
save_path = fig_path + f"{folder}/"
freq_range = [15, 30]
white_ratio = 0
win_sec = 4
nperseg = int(win_sec * srate)


slopes = np.arange(0, 4.5, .5)

# Oscillation
freq_osc = [14, 18, 27]
amp = [1, 1, 1]
width = [4, 2, 4]

slopes = np.array([.8, 1.2])


# hset_max = srate / 4 / freq_range[1]
hset_max = 9.9

irasa_params = {"sf": srate, "ch_names": slopes, "win_sec": win_sec,
                "kwargs_welch": {'average': 'mean'},
                "hset": np.linspace(1.1, hset_max, num=15)}

signals = osc_signals(samples, slopes, freq_osc, amp, width=width)


# highpass filter
sos = sig.butter(10, 1, btype="hp", fs=srate, output='sos')
signals = sig.sosfilt(sos, signals)
freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
save_name = (f"{freq_name}_beta_width={width}_"
             f"hset_max={hset_max:.2f}.pdf")
add_title = f"beta width={width} SDs, hmax={hset_max:.2f}"
freq, noise_psds = psds_pink(signals, srate, nperseg)
IRASA = irasa(data=signals, band=freq_range, **irasa_params)

plot_all(freq, noise_psds, freq, noise_psds, IRASA, slopes, freq_range,
         white_ratio,
         plot_osc=True,
         save_path=save_path, save_name=save_name, add_title=add_title)
