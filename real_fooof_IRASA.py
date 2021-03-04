"""Compare fooof with IRASA on real data."""
import numpy as np
import scipy.signal as sig
from scipy.stats import norm
import matplotlib.pyplot as plt
# import matplotlib.ticker
from fooof import FOOOF, FOOOFGroup
from fooof.sim.gen import gen_aperiodic
import yasa  # IRASA
# from pathlib import Path
# import pandas as pd
# import mne
from helper import load_fif
from helper import load_psd

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
        return noise # (noise - noise.mean()) / noise.std()
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
    if noises.ndim == 1:
        freq, psd = sig.welch(noises, fs=srate, nperseg=nperseg)
        # divide by max value to have the same offset at 2Hz
        if normalize:
            return freq, psd / psd.max()
        else:
            return freq, psd
    elif noises.ndim == 2:
        noise_psds = []
        for i in range(noises.shape[0]):
            freq, psd = sig.welch(noises[i], fs=srate, nperseg=nperseg)
            # divide by max value to have the same offset at 2Hz
            if normalize:
                noise_psds.append(psd / psd.max())
            else:
                noise_psds.append(psd)
        return freq, np.array(noise_psds)
    else:
        raise "Dimension must be 1 or 2"


# %% LOAD DATA
raw_conds = load_fif()
freqs, PSD_on, PSD_off = load_psd()

# select data
subj = 13
ch = 8

signal_on = raw_conds[0][subj]  # on
signal_off = raw_conds[1][subj]  # off
psd_on = PSD_on[subj, ch]
psd_off = PSD_off[subj, ch]
signal_on = signal_on.get_data()[ch]
signal_off = signal_off.get_data()[ch]
# %% PARAMETERS

# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slopes = np.arange(0, 4.5, .5)

# WELCH
win_sec = 1
nperseg = int(win_sec * srate)

# Path
fig_path = "plots/"

# Colors
c_on = "#E66101"
c_off = "#5E3C99"

# %% Simulate PSD
slope_truth_on = 1.15
freq_osc_on = [2, 4, 6.9, 14, 27]
amp_on = [8, 0.3, 0.4, .35, .4]
width_on = [0.01, 2, 1.6, 4, 4]

slope_truth_off = .95
freq_osc_off = [2, 6, 14, 27]
amp_off = [12.5, .4, .9, .6]
width_off = [.001, .7, 2.4, 5]

signal_sim_on = osc_signals(samples, slope_truth_on,
                            freq_osc_on, amp_on, width=width_on)
signal_sim_off = osc_signals(samples, slope_truth_off,
                             freq_osc_off, amp_off, width=width_off)
signal_sim_on_ap = osc_signals(samples, slope_truth_on,
                               None, None, width=None)
signal_sim_off_ap = osc_signals(samples, slope_truth_off,
                                None, None, width=None)
signal_sim_on_osc = osc_signals(samples, 0,
                                freq_osc_on, amp_on, width=width_on)
signal_sim_off_osc = osc_signals(samples, 0,
                                 freq_osc_off, amp_off, width=width_off)

# Calc PSD
frequencies, psd_sim_on = psds_pink(signal_sim_on, srate, nperseg)
frequencies, psd_sim_off = psds_pink(signal_sim_off, srate, nperseg)

frequencies, psd_sim_on_ap = psds_pink(signal_sim_on_ap, srate, nperseg)
frequencies, psd_sim_off_ap = psds_pink(signal_sim_off_ap, srate, nperseg)

frequencies, psd_sim_on_osc = psds_pink(signal_sim_on_osc, srate, nperseg)
frequencies, psd_sim_off_osc = psds_pink(signal_sim_off_osc, srate, nperseg)

# make freqs equivalent to loaded real data
pre_mask = (frequencies >= 1) & (frequencies <= 600)

frequencies = frequencies[pre_mask]
psd_sim_on = psd_sim_on[pre_mask]
psd_sim_off = psd_sim_off[pre_mask]
psd_sim_on_ap = psd_sim_on_ap[pre_mask]
psd_sim_off_ap = psd_sim_off_ap[pre_mask]
psd_sim_on_osc = psd_sim_on_osc[pre_mask]
psd_sim_off_osc = psd_sim_off_osc[pre_mask]


# %% Tune Parameters

for i in range(9):

        fit_params = {
            0: {"IRASA": {"band": [1, 45],
                          "sf": srate, "win_sec": 0.5,
                          "hset": np.linspace(1.1, 1.9, num=5),
                          "kwargs_welch": {'average': 'median'}},
                "fooof": {"peak_width_limits": (1, 12), "peak_threshold": 1,
                          "max_n_peaks": np.inf, "min_peak_height": 0,
                          "aperiodic_mode": "fixed"}},

            1: {"IRASA": {"band": [2, 45],
                          "sf": srate, "win_sec": 0.5,
                          "hset": np.linspace(1.1, 1.9, num=5),
                          "kwargs_welch": {'average': 'median'}},
                "fooof": {"peak_width_limits": (1, 12), "peak_threshold": 1,
                          "max_n_peaks": np.inf, "min_peak_height": 0,
                          "aperiodic_mode": "fixed"}},

            2: {"IRASA": {"band": [5, 45],
                          "sf": srate, "win_sec": 1,
                          "hset": np.linspace(1.1, 2.9, num=5),
                          "kwargs_welch": {'average': 'median'}},
                "fooof": {"peak_width_limits": (1, 12), "peak_threshold": 1,
                          "max_n_peaks": np.inf, "min_peak_height": 0,
                          "aperiodic_mode": "fixed"}},

            3: {"IRASA": {"band": [10, 45],
                          "sf": srate, "win_sec": 1,
                          "hset": np.linspace(1.1, 2.9, num=5),
                          "kwargs_welch": {'average': 'median'}},
                "fooof": {"peak_width_limits": (1, 12), "peak_threshold": 1,
                          "max_n_peaks": np.inf, "min_peak_height": 0,
                          "aperiodic_mode": "fixed"}},

            4: {"IRASA": {"band": [15, 45],
                          "sf": srate, "win_sec": 1,
                          "hset": np.linspace(1.1, 2.9, num=5),
                          "kwargs_welch": {'average': 'median'}},
                "fooof": {"peak_width_limits": (1, 12), "peak_threshold": 1,
                          "max_n_peaks": np.inf, "min_peak_height": 0,
                          "aperiodic_mode": "fixed"}},

            5: {"IRASA": {"band": [20, 45],
                          "sf": srate, "win_sec": 1,
                          "hset": np.linspace(1.1, 2.9, num=7),
                          "kwargs_welch": {'average': 'median'}},
                "fooof": {"peak_width_limits": (1, 12), "peak_threshold": 1,
                          "max_n_peaks": np.inf, "min_peak_height": 0,
                          "aperiodic_mode": "fixed"}},

            6: {"IRASA": {"band": [30, 45],
                          "sf": srate, "win_sec": 1,
                          "hset": np.linspace(1.1, 2.9, num=5),
                          "kwargs_welch": {'average': 'median'}},
                "fooof": {"peak_width_limits": (1, 12), "peak_threshold": 1,
                          "max_n_peaks": np.inf, "min_peak_height": 0,
                          "aperiodic_mode": "fixed"}},

            7: {"IRASA": {"band": [35, 45],
                          "sf": srate, "win_sec": 1,
                          "hset": np.linspace(1.1, 1.9, num=5),
                          "kwargs_welch": {'average': 'median'}},
                "fooof": {"peak_width_limits": (1, 12), "peak_threshold": 1,
                          "max_n_peaks": np.inf, "min_peak_height": 0,
                          "aperiodic_mode": "fixed"}},

            8: {"IRASA": {"band": [40, 60],
                          "sf": srate, "win_sec": .5,
                          "hset": np.linspace(1.1, 3.9, num=5),
                          "kwargs_welch": {'average': 'median'}},
                "fooof": {"peak_width_limits": (1, 12), "peak_threshold": 1,
                          "max_n_peaks": np.inf, "min_peak_height": 0,
                          "aperiodic_mode": "fixed"}}
                        }

        # if peak is wide, num should be smaller!
        # if freq range is narrow, haset_max should be smaller!
        # small win_sec enables better 1/f fitting but worse oscillation extraction
        # fooof needs little tuning

        freq_range = fit_params[i]["IRASA"]["band"]

        # Normalize power offset for plotting
        psd_on = psd_on / psd_on[0]
        psd_off = psd_off / (psd_off[0] * 10)  # different offset for off
        psd_sim_on = psd_sim_on / psd_sim_on[0]
        psd_sim_off = psd_sim_off / (psd_sim_off[0] * 10)
        psd_sim_on_ap = psd_sim_on_ap / psd_sim_on_ap[0]
        psd_sim_off_ap = psd_sim_off_ap / (psd_sim_off_ap[0] * 10)
        

        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        
        
        fooof_params = fit_params[i]["fooof"]
        irasa_params = fit_params[i]["IRASA"]

        irasa_params = {"band": freq_range,
                        "sf": srate, "win_sec": 1,
                        "hset": np.linspace(1.1, 1.9, num=5),
                        "kwargs_welch": {'average': 'median'}}
        
        # % Calc and Plot
        
        # IRASA
        IRASA_real_on = yasa.irasa(data=signal_on, **irasa_params)
        IRASA_real_off = yasa.irasa(data=signal_off, **irasa_params)
        IRASA_sim_on = yasa.irasa(data=signal_sim_on, **irasa_params)
        IRASA_sim_off = yasa.irasa(data=signal_sim_off, **irasa_params)
        
        freq, aperiodic_on, osc_on, params_on = IRASA_real_on
        freq, aperiodic_off, osc_off, params_off = IRASA_real_off
        freq, aperiodic_sim_on, osc_sim_on, params_sim_on = IRASA_sim_on
        freq, aperiodic_sim_off, osc_sim_off, params_sim_off = IRASA_sim_off
        
        # Normalize power offset
        sum_on = aperiodic_on[0] + osc_on[0]
        sum_on_max = sum_on[0]
        sum_on /= sum_on_max
        sum_off = aperiodic_off[0] + osc_off[0]
        sum_off_max = sum_off[0]
        sum_off /= (sum_off_max * 10)
        sum_sim_on = aperiodic_sim_on[0] + osc_sim_on[0]
        sum_sim_on_max = sum_sim_on[0]
        sum_sim_on /= sum_sim_on_max
        sum_sim_off = aperiodic_sim_off[0] + osc_sim_off[0]
        sum_sim_off_max = sum_sim_off[0]
        sum_sim_off /= (sum_sim_off_max * 10)
        
        # IRASA sim and real needs correct scaling
        scale = sum_on_max / sum_sim_on_max
        osc_on = osc_on[0]# / osc_on[0].max()
        osc_off = osc_off[0] - .015  # subtract for offset diff
        osc_sim_on = osc_sim_on[0] * scale
        osc_sim_off = osc_sim_off[0] * scale - .015
        
        aperiodic_on = aperiodic_on[0] / sum_on_max
        aperiodic_sim_on = aperiodic_sim_on[0] / sum_sim_on_max
        aperiodic_off = aperiodic_off[0] / (sum_off_max * 10)
        aperiodic_sim_off = aperiodic_sim_off[0] / (sum_sim_off_max * 10)
        
        # Extract Fit Params
        offset_on = params_on["Intercept"].iloc[0]
        slope_on = -params_on["Slope"].iloc[0]
        offset_off = params_off["Intercept"].iloc[0]
        slope_off = -params_off["Slope"].iloc[0]
        offset_sim_on = params_sim_on["Intercept"].iloc[0]
        slope_sim_on = -params_sim_on["Slope"].iloc[0]
        offset_sim_off = params_sim_off["Intercept"].iloc[0]
        slope_sim_off = -params_sim_off["Slope"].iloc[0]
        
        # Calc fit
        freq_fit = np.r_[0, freq]  # we need to start at 0
        fit_on = 10**offset_on / freq_fit ** slope_on
        fit_off = 10**offset_off / freq_fit ** slope_off
        fit_sim_on = 10**offset_sim_on / freq_fit ** slope_sim_on
        fit_sim_off = 10**offset_sim_off / freq_fit ** slope_sim_off
        
        # Normalize Fit
        fit_on /= fit_on[1]
        fit_off /= fit_off[1] * 10
        fit_sim_on /= fit_sim_on[1]
        fit_sim_off /= fit_sim_off[1] * 10
        
        # Extract error params
        r_squ_on = params_on["R^2"].iloc[0]
        r_squ_off = params_off["R^2"].iloc[0]
        r_squ_sim_on = params_sim_on["R^2"].iloc[0]
        r_squ_sim_off = params_sim_off["R^2"].iloc[0]
        
        # Calc mean error
        R2_mean_I = np.mean([r_squ_on, r_squ_off, r_squ_sim_on, r_squ_sim_off])
        
        # Labels
        label_on = f"Real on 1/f {slope_on:.2f}"
        label_off = f"Real off 1/f {slope_off:.2f}"
        label_sim_on = f"Sim on 1/f {slope_sim_on:.2f}"
        label_sim_off = f"Sim off 1/f {slope_sim_off:.2f}"
        
        # % Plot Both
        
        # IRASA Plot arguments
        real_on = (freqs[mask], psd_on[mask])
        kwgs_on = {"c": c_on, "label": "Real On"}
        real_off = (freqs[mask], psd_off[mask])
        kwgs_off = {"c": c_off, "label": "Real Off"}
        
        sim_on = (freqs[mask], psd_sim_on[mask], "--")
        kwgs_sim_on = {"c": c_on, "label": f"Sim On 1/f={slope_truth_on}"}
        sim_on_ap = (freqs[mask], psd_sim_on_ap[mask], "b--")
        sim_on_osc = (freqs[mask], psd_sim_on_osc[mask], "--")
        kwgs_sim_on_osc = {"c": c_on, "label": "Sim osc ground truth On"}
        
        sim_off = (freqs[mask], psd_sim_off[mask], "--")
        kwgs_sim_off = {"c": c_off, "label": f"Sim Off 1/f={slope_truth_off}"}
        sim_off_ap = (freqs[mask], psd_sim_off_ap[mask], "b--")
        kwgs_sim_off_ap = {"label": "Sim Aperiodic"}
        sim_off_osc = (freqs[mask], psd_sim_off_osc[mask]-4e-9, "--")
        kwgs_sim_off_osc = {"c": c_off, "label": "Sim osc ground truth Off"}
        
        #
        
        fig, axes = plt.subplots(3, 3, figsize=[20, 15])
        
        ax = axes[0, 0]
        ax.loglog(*real_on, **kwgs_on)
        ax.loglog(*real_off, **kwgs_off)
        ax.loglog(*sim_on, **kwgs_sim_on)
        ax.loglog(*sim_on_ap)
        ax.loglog(*sim_off, **kwgs_sim_off)
        ax.loglog(*sim_off_ap, **kwgs_sim_off_ap)
        
        ax.set_ylabel("Log Power a.u.")
        ax.set_xlabel("log(Frequency)")
        ax.set_title("Original PSD")
        ax.legend()
        
        ax = axes[1, 0]
        ax.semilogy(*real_on, **kwgs_on)
        ax.semilogy(*real_off, **kwgs_off)
        ax.semilogy(*sim_on, **kwgs_sim_on)
        ax.semilogy(*sim_on_ap)
        ax.semilogy(*sim_off, **kwgs_sim_off)
        ax.semilogy(*sim_off_ap, **kwgs_sim_off_ap)
        
        ax.set_ylabel("Log Power a.u.")
        ax.set_xticklabels("")
        ax.legend()
        
        ax = axes[2, 0]
        ax.plot(*sim_on_osc, **kwgs_sim_on_osc)
        ax.plot(*sim_off_osc, **kwgs_sim_off_osc)
        
        ax.set_title(f"Welch: {win_sec}s")
        ax.set_ylabel("Power a.u.")
        ax.set_xlabel("Frequency")
        ax.legend()
        
        # fooof
        ax = axes[0, 1]
        fm_on = FOOOF(**fooof_params)  # Init fooof
        fm_on.fit(freqs, psd_on, freq_range)
        fm_on.plot(plt_log=True, ax=ax,
                   model_kwargs={"color": c_on, "alpha": 1, "linewidth": 2},
                   data_kwargs={"alpha": 0})
        exponent_on = fm_on.get_params('aperiodic_params', 'exponent')
        r_squ_on = fm_on.get_params('r_squared')
        
        
        fm_off = FOOOF(**fooof_params)  # Init fooof
        fm_off.fit(freqs, psd_off, freq_range)
        fm_off.plot(plt_log=True, ax=ax,
                    model_kwargs={"color": c_off, "alpha": 1, "linewidth": 2},
                    data_kwargs={"alpha": 0})
        exponent_off = fm_off.get_params('aperiodic_params', 'exponent')
        r_squ_off = fm_off.get_params('r_squared')
        
        fm_on_sim = FOOOF(**fooof_params)  # Init fooof
        fm_on_sim.fit(freqs, psd_sim_on, freq_range)
        fm_on_sim.plot(plt_log=True, ax=ax,
                       model_kwargs={"color": c_on, "linestyle": "--", "alpha": 1,
                                     "linewidth": 2},
                       data_kwargs={"alpha": 0})
        exponent_sim_on = fm_on_sim.get_params('aperiodic_params', 'exponent')
        r_squ_sim_on = fm_on_sim.get_params('r_squared')
        
        fm_off_sim = FOOOF(**fooof_params)  # Init fooof
        fm_off_sim.fit(freqs, psd_sim_off, freq_range)
        fm_off_sim.plot(plt_log=True, ax=ax,
                        model_kwargs={"color": c_off, "linestyle": "--", "alpha": 1,
                                      "linewidth": 2},
                        data_kwargs={"alpha": 0})
        exponent_sim_off = fm_off_sim.get_params('aperiodic_params', 'exponent')
        r_squ_sim_off = fm_off_sim.get_params('r_squared')
        
        ax.set_ylabel("")
        R2_mean_F = np.mean([r_squ_on, r_squ_off, r_squ_sim_on, r_squ_sim_off])
        ax.set_title(f"fooof R^2 mean: {R2_mean_F:.2f}")
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[1::3] + [handles[-1]]
        label_on = f"Real On 1/f={exponent_on:.2f}"
        label_off = f"Real Off 1/f={exponent_off:.2f}"
        label_sim_on = f"Sim On 1/f={exponent_sim_on:.2f}"
        label_sim_off = f"Sim Off 1/f={exponent_sim_off:.2f}"
        labels = [label_on, label_off, label_sim_on, label_sim_off, "Aperiodic Fit"]
        ax.legend(handles, labels)
        ax.grid(False)

        ax = axes[1, 1]
        error_fooof_sim = (abs(slope_truth_on - exponent_sim_on) +
                           abs(slope_truth_off - exponent_sim_off))
        error_fooof = (abs(slope_truth_on - exponent_on) +
                       abs(slope_truth_off - exponent_off))
        ax.set_title(f"Fooof error sim: {error_fooof_sim:.2f}, "
                     f"real: {error_fooof:.2f}")
        fm_on.plot(plt_log=False, ax=ax,
                   model_kwargs={"color": c_on, "linestyle": "-", "alpha": 1,
                                 "linewidth": 2},
                   data_kwargs={"alpha": 0})
        fm_off.plot(plt_log=False, ax=ax,
                    model_kwargs={"color": c_off, "linestyle": "-", "alpha": 1,
                                  "linewidth": 2},
                    data_kwargs={"alpha": 0})
        fm_on_sim.plot(plt_log=False, ax=ax,
                       model_kwargs={"color": c_on, "linestyle": "--", "alpha": 1,
                                     "linewidth": 2},
                       data_kwargs={"alpha": 0})
        fm_off_sim.plot(plt_log=False, ax=ax,
                        model_kwargs={"color": c_off, "linestyle": "--", "alpha": 1,
                                      "linewidth": 2},
                        data_kwargs={"alpha": 0})
        
        ax.grid(False)
        ax.set_xticklabels("")
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.legend(handles, labels)
        
        ax = axes[2, 1]
        ap_fit_on = gen_aperiodic(fm_on.freqs, fm_on.aperiodic_params_)
        fooof_osc_on = fm_on.power_spectrum - ap_fit_on
        ax.plot(fm_on.freqs, fooof_osc_on, c=c_on, label="osc on")
        
        ap_fit_off = gen_aperiodic(fm_off.freqs, fm_off.aperiodic_params_)
        fooof_osc_off = fm_off.power_spectrum - ap_fit_off
        ax.plot(fm_off.freqs, fooof_osc_off-.8, c=c_off, label="osc off")
        
        ap_fit_sim_on = gen_aperiodic(fm_on_sim.freqs, fm_on_sim.aperiodic_params_)
        fooof_osc_sim_on = fm_on_sim.power_spectrum - ap_fit_sim_on
        ax.plot(fm_on_sim.freqs, fooof_osc_sim_on, "--", c=c_on, label="osc sim on")
        ap_fit_sim_off = gen_aperiodic(fm_off_sim.freqs, fm_off_sim.aperiodic_params_)
        fooof_osc_sim_off = fm_off_sim.power_spectrum - ap_fit_sim_off
        ax.plot(fm_off_sim.freqs, fooof_osc_sim_off-.8, "--", c=c_off,
                label="osc sim off")
        
        peak_width_min = fooof_params["peak_width_limits"][0]
        ax.set_title(f"Welch: {win_sec}s, min peak width: {peak_width_min}Hz")
        ax.set_xlabel('Frequency')
        ax.legend()
        
        # IRASA
        ax = axes[0, 2]
        
        # Labels
        label_on = f"Real on 1/f {slope_on:.2f}"
        label_off = f"Real off 1/f {slope_off:.2f}"
        label_sim_on = f"Sim on 1/f {slope_sim_on:.2f}"
        label_sim_off = f"Sim off 1/f {slope_sim_off:.2f}"
        
        ir_on = (freq, sum_on, c_on)
        ir_off = (freq, sum_off, c_off)
        ir_sim_on = (freq, sum_sim_on, "--")
        ir_sim_off = (freq, sum_sim_off, "--")
        ir_ap_on = (freq, aperiodic_on, "darkgreen")
        ir_ap_off = (freq, aperiodic_off, "darkgreen")
        ir_ap_sim_on = (freq, aperiodic_sim_on, "--")
        ir_ap_sim_off = (freq, aperiodic_sim_off, "--")
        
        ax.loglog(*ir_on, label=label_on)
        ax.loglog(*ir_off, label=label_off)
        ax.loglog(*ir_sim_on, c=c_on, label=label_sim_on)
        ax.loglog(*ir_sim_off, c=c_off, label=label_sim_off)
        
        # Fits
        p_fit_on = (freq_fit, fit_on, "b--")
        p_fit_off = (freq_fit, fit_off, "b--")
        p_fit_sim_on = (freq_fit, fit_sim_on, "b--")
        p_fit_sim_off = (freq_fit, fit_sim_off, "b--")
        
        ax.loglog(*ir_ap_on, label="Aperiodic")
        ax.loglog(*ir_ap_off)
        ax.loglog(*ir_ap_sim_on, c="darkgreen")
        ax.loglog(*ir_ap_sim_off, c="darkgreen")
        
        ax.loglog(*p_fit_on, label="Aperiodic Fit")
        ax.loglog(*p_fit_off)
        ax.loglog(*p_fit_sim_on)
        ax.loglog(*p_fit_sim_off)
        
        ax.set_title(f"IRASA R^2 mean: {R2_mean_I:.2f}")
        ax.set_xlabel("log(Frequency)")
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        
        
        ax = axes[1, 2]
        error_IRASA_sim = (abs(slope_truth_on - slope_sim_on) +
                           abs(slope_truth_off - slope_sim_off))
        error_IRASA = (abs(slope_truth_on - slope_on) +
                       abs(slope_truth_off - slope_off))
        ax.set_title(f"IRASA error sim: {error_IRASA_sim:.2f}, "
                     f"real: {error_IRASA:.2f}")
        
        ax.semilogy(*ir_on)
        ax.semilogy(*ir_ap_on)
        ax.semilogy(*ir_off)
        ax.semilogy(*ir_ap_off)
        ax.semilogy(*ir_sim_on, c=c_on)
        ax.semilogy(*ir_ap_sim_on, c="darkgreen")
        ax.semilogy(*ir_sim_off, c=c_off)
        ax.semilogy(*ir_ap_sim_off, c="darkgreen")
        ax.semilogy(*p_fit_on)
        ax.semilogy(*p_fit_off)
        ax.semilogy(*p_fit_sim_on)
        ax.semilogy(*p_fit_sim_off)
        ax.set_xticklabels("")
        ax.legend(handles, labels)
        
        
        ax = axes[2, 2]
        ax.plot(freq, osc_on, c_on, label="osc on")
        ax.plot(freq, osc_off, c_off, label="osc off")
        ax.plot(freq, osc_sim_on, "--", c=c_on, label="osc sim")
        ax.plot(freq, osc_sim_off, "--", c=c_off, label="osc sim")
        
        win_sec_i = irasa_params["win_sec"]
        hset = irasa_params["hset"]
        hset_max = hset[-1]
        num = hset.size
        ax.set_title(f"Welch: {win_sec_i}s, hmax={hset_max}, num={num}")
        ax.set_xlabel("Frequency")
        ax.legend()
        
        freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
        plt.suptitle(f"Fit Range: {freq_name}", fontsize=20)
        for ax in axes.flatten():
            ax.set_yticklabels("")
        plt.tight_layout()
        plt.show()
        
        
        # save errors as df and plot

        # Questions: can IRASA deal with bad segments? -> seems like it? reall? why
        # such a big difference between average=mean and median?
        # IRASA: knickt ab bei niedrigen frequenzen -> Artefakt. Option1 : 
            # wähle hset_max niedrig -> nicht alle oszis werden entdeckt
            # wähle höhere minimum frequenz bereich
        # =============================================================================
