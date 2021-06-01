"""
Very broad peak widths require very large resampling factors.

Panel a, b, c:
    - same as panel a Fig 4 for increasing peak widths!
    - include aperiodic components

d, e, f:
    - real data with small, medium, and large peak widths and different
    resampling factors
    - large peak: use LFP data of Esther
    - medium peak: use MEG source data of Esther
    - small peak:
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig
from pathlib import Path
import mne
from fooof import FOOOF
# from fooof.sim.gen import gen_aperiodic
import matplotlib.gridspec as gridspec
from noise_helper import irasa
try:
    from tqdm import trange
except ImportError:
    trange = range


def osc_signals(slope, periodic_params=None, nlv=None, highpass=True,
                srate=2400, duration=180, seed=1):
    """
    Generate colored noise with optionally added oscillations.

    Parameters
    ----------
    slope : float, optional
        Aperiodic 1/f exponent. The default is 1.
    periodic_params : list of tuples, optional
        Oscillations parameters as list of tuples in form
        [(frequency, amplitude, width), (frequency, amplitude, width)] for
        two oscillations.
        The default is None.
    nlv : float, optional
        Level of white noise. The default is None.
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
            amp_dist = sp.stats.norm(freq_osc, width).pdf(freqs)
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

    # Highpass filter
    if highpass:
        sos = sig.butter(4, 1, btype="hp", fs=srate, output='sos')
        noise = sig.sosfilt(sos, noise)
        noise_osc = sig.sosfilt(sos, noise_osc)

    return noise, noise_osc


def calc_error(signal):
    """Fit IRASA and subtract ground truth to obtain fitting error."""
    fit_errors = []
    for i in trange(len(lower_fitting_borders)):
        freq_range = (lower_fitting_borders[i], upper_fitting_border)
        _, _, _, params = irasa(data=signal, band=freq_range, sf=srate)
        exp = -params["Slope"][0]
        error = np.abs(toy_slope - exp)
        fit_errors.append(error)
    return fit_errors


# %% PARAMETERS

# Signal params
srate = 2400
nperseg = srate
welch_params = dict(fs=srate, nperseg=nperseg)

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig5_IRASA_PeakWidth.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# File names
path = "../data/Fig5/"

# Colors

# a)
c_sim = "k"
c_error = "r"
c_noise = "darkgray"

c_range1 = "b"
c_range2 = "g"
c_range3 = "y"

c_ap = "grey"

# b)
c_real = "purple"

# c_fit3 = "lime"

# c)
c_fooof = "deepskyblue"
c_IRASA1 = "C1"
c_IRASA2 = "C2"
c_IRASA3 = "orangered"

lw = 2

# %% a Sim Toy Signal with Three Oscillations and Fit

# fit in all frequency ranges from 1 to 80...
lower_fitting_borders = range(1, 80)
# ... to 100 Hz
upper_fitting_border = 100

# Oscillations parameters:
toy_slope = 2
freq1, freq2, freq3 = 5, 15, 35  # Hz
amp = 1
width = 1

periodic_params_a = [(freq1, amp*2, width*.2),
                     (freq2, amp*.4, width*.25),
                     (freq3, amp*.35, width*1.8)]

periodic_params_b = [(freq1, amp*2.8, width*.42),
                     (freq2, amp*1, width*.84),
                     (freq3, amp*.6, width*2)]

periodic_params_c = [(freq1, amp*4.4, width*.8),
                     (freq2, amp*1.7, width*1.2),
                     (freq3, amp*.7, width*2.5)]
# Sim Toy Signal
_, toy_signal_a = osc_signals(toy_slope, periodic_params=periodic_params_a,
                              highpass=False)
_, toy_signal_b = osc_signals(toy_slope, periodic_params=periodic_params_b,
                              highpass=False)
_, toy_signal_c = osc_signals(toy_slope, periodic_params=periodic_params_c,
                              highpass=False)

freq_a, toy_psd_a = sig.welch(toy_signal_a, **welch_params)
freq_b, toy_psd_b = sig.welch(toy_signal_b, **welch_params)
freq_c, toy_psd_c = sig.welch(toy_signal_c, **welch_params)

# Filter 1-100Hz
filt_a = (freq_a <= 100)
freq_a = freq_a[filt_a]
toy_psd_a = toy_psd_a[filt_a]
toy_psd_b = toy_psd_b[filt_a]
toy_psd_c = toy_psd_c[filt_a]

# %% Calc Aperiodic Component for largest range

freq_range = (lower_fitting_borders[0], upper_fitting_border)
freq0, psd_aperiodic_a, _, _ = irasa(toy_signal_a, band=freq_range, sf=srate)
_, psd_aperiodic_b, _, _ = irasa(toy_signal_b, band=freq_range, sf=srate)
_, psd_aperiodic_c, _, _ = irasa(toy_signal_c, band=freq_range, sf=srate)

psd_aperiodic_a = psd_aperiodic_a[0]
psd_aperiodic_b = psd_aperiodic_b[0]
psd_aperiodic_c = psd_aperiodic_c[0]
# %% a IRASA (takes very long)

fit_errors_a = calc_error(toy_signal_a)
fit_errors_b = calc_error(toy_signal_b)
fit_errors_c = calc_error(toy_signal_c)

error_plot_a = (lower_fitting_borders, fit_errors_a, c_error)
error_plot_b = (lower_fitting_borders, fit_errors_b, c_error)
error_plot_c = (lower_fitting_borders, fit_errors_c, c_error)


# %% C Real data

srate = 2400
nperseg = srate
welch_params = dict(fs=srate, nperseg=nperseg)

highpass = .3  # Hz
lowpass = 600  # Hz

fname_MEG = "Subj016_ON_1_raw.fif"
fname_LFP = "Subj016_OFF_001_STN_r+s_raw.fif"

sub_MEG = mne.io.read_raw_fif(path + fname_MEG, preload=True)
sub_LFP = mne.io.read_raw_fif(path + fname_LFP, preload=True)

# ch2 = "MEG2621" # beta peak overlay: 0433
small_grad = "MEG0913"
med_mag = "MEG0113"
large_LFP = "R2-R3"

channels = [small_grad, med_mag]

sub_MEG.pick_channels(channels, ordered=True)
sub_LFP.pick_channels([large_LFP])

# Convert to numpy
MEG_raw = sub_MEG.get_data()
LFP_raw = sub_LFP.get_data()[0]
LFP_raw *= 1e6  # convert V to uV


freq, spec_MEG = sig.welch(MEG_raw, **welch_params)
freq, spec_LFP = sig.welch(LFP_raw, **welch_params)


filt = (freq <= 600)
freq_filt = freq[filt]
spec_GRAD = spec_MEG[0, filt]
spec_MAG = spec_MEG[1, filt]


plot_small = (freq_filt, spec_GRAD, c_real)
plot_med = (freq_filt, spec_MAG, c_real)
plot_large = (freq, spec_LFP, c_real)

# plot_small_low = (freq, spec_MEG[0]/100, c_real)
plot_med_low = (freq_filt, spec_MAG/100, c_real)
plot_large_low = (freq, spec_LFP/100, c_real)

# plot_small_lower = (freq, spec_MEG[0]/100, c_real)
# plot_med_lower = (freq, spec_MEG[1]/100, c_real)
plot_large_lower = (freq, spec_LFP/10000, c_real)

# %% C Apply fooof to determine peak width

real_s = FOOOF(max_n_peaks=2, verbose=False)
real_m = FOOOF(max_n_peaks=1, verbose=False, peak_width_limits=(0.5, 150))
# real_m12 = FOOOF(max_n_peaks=2, verbose=False, peak_width_limits=(0.5, 150))
real_l = FOOOF(max_n_peaks=1, verbose=False, peak_width_limits=(.5, 150))

real_s.fit(freq, spec_MEG[0], (3, 20))
real_m.fit(freq, spec_MEG[1], (1, 100))
# real_m12.fit(freq, spec_MEG[1], (1, 100))
real_l.fit(freq, spec_LFP, (10, 150))

# real_m.plot(plt_log=True)

freq_real_s, pow_real_s, bw_real_s = real_s.peak_params_[0]
freq_real_m, pow_real_m, bw_real_m = real_m.peak_params_[0]
# freq_real_m1, pow_real_m1, bw_real_m1 = real_m12.peak_params_[0]
# freq_real_m2, pow_real_m2, bw_real_m2 = real_m12.peak_params_[1]
freq_real_l, pow_real_l, bw_real_l = real_l.peak_params_[0]

# %% C Calc IRASA
h_max1 = 2
h_max2 = 25
h_max3 = 45

# doesn't matter that shorter band makes more sence, this is topic of fig4
band = (1, 100)

band_h1 = (highpass * h_max1, lowpass / h_max1)
band_h2 = (highpass * h_max2, lowpass / h_max2)
band_h3 = (highpass * h_max3, lowpass / h_max3)

N_h = 16
# N_h = 5  # increase in the end

irasa_params1 = dict(sf=srate, band=band_h1, hset=np.linspace(1.1, h_max1, N_h))
irasa_params2 = dict(sf=srate, band=band_h2, hset=np.linspace(1.1, h_max2, N_h))
irasa_params3 = dict(sf=srate, band=band_h3, hset=np.linspace(1.1, h_max3, N_h))

IRASA_small_h1 = irasa(MEG_raw[0], **irasa_params1)
IRASA_med_h1 = irasa(MEG_raw[1], **irasa_params1)
IRASA_large_h1 = irasa(LFP_raw, **irasa_params1)
IRASA_m_h2 = irasa(MEG_raw[1], **irasa_params2)
IRASA_l_h2 = irasa(LFP_raw, **irasa_params2)
IRASA_l_h3 = irasa(LFP_raw, **irasa_params3)


freq_I_h1, ap_small_h1, per_small_h1, params_small_h1 = IRASA_small_h1
_, ap_med_h1, per_med_h1, params_med_h1 = IRASA_med_h1
_, ap_large_h1, per_large_h1, params_large_h1 = IRASA_large_h1
freq_I_h2, ap_med_h2, per_med_h2, params_med_h2 = IRASA_m_h2
freq_I_h2, ap_large_h2, per_large_h2, params_large_h2 = IRASA_l_h2
freq_I_h3, ap_large_h3, per_large_h3, params_large_h3 = IRASA_l_h3


plot_ap_small_h1 = (freq_I_h1, ap_small_h1[0], c_IRASA1)
plot_ap_med_h1 = (freq_I_h1, ap_med_h1[0], c_IRASA1)
plot_ap_large_h1 = (freq_I_h1, ap_large_h1[0], c_IRASA1)
plot_ap_med_h2 = (freq_I_h2, ap_med_h2[0]/100, c_IRASA2)
plot_ap_large_h2 = (freq_I_h2, ap_large_h2[0]/100, c_IRASA2)
plot_ap_large_h3 = (freq_I_h3, ap_large_h3[0]/10000, c_IRASA3)


# Show what happens for larger freq ranges
irasa_params2["band"] = band_h1
irasa_params3["band"] = band_h1

IRASA_med_h1_long = irasa(MEG_raw[1], **irasa_params1)
IRASA_large_h1_long = irasa(LFP_raw, **irasa_params1)
IRASA_m_h2_long = irasa(MEG_raw[1], **irasa_params2)
IRASA_l_h2_long = irasa(LFP_raw, **irasa_params2)
IRASA_l_h3_long = irasa(LFP_raw, **irasa_params3)

freq_I_long, ap_med_h1_long, per_med_h1_long, params_med_h1_long = IRASA_med_h1_long
_, ap_large_h1_long, per_large_h1_long, params_large_h1_long = IRASA_large_h1_long
_, ap_med_h2_long, per_med_h2_long, params_med_h2_long = IRASA_m_h2_long
_, ap_large_h2_long, per_large_h2_long, params_large_h2_long = IRASA_l_h2_long
_, ap_large_h3_long, per_large_h3_long, params_large_h3_long = IRASA_l_h3_long

plot_ap_med_h1_long = (freq_I_long, ap_med_h1_long[0], c_IRASA1)
plot_ap_large_h1_long = (freq_I_long, ap_large_h1_long[0], c_IRASA1)
plot_ap_med_h2_long = (freq_I_long, ap_med_h2_long[0]/100, c_IRASA2)
plot_ap_large_h2_long = (freq_I_long, ap_large_h2_long[0]/100, c_IRASA2)
plot_ap_large_h3_long = (freq_I_long, ap_large_h3_long[0]/10000, c_IRASA3)


# =============================================================================
# # Extract fits
# small_h1_off = params_small_h1["Intercept"][0]
# small_h1_slope = -params_small_h1["Slope"][0]
# med_h1_off = params_med_h1["Intercept"][0]
# med_h2_off = params_med_h2["Intercept"][0]
# med_h1_slope = -params_med_h1["Slope"][0]
# med_h2_slope = -params_med_h2["Slope"][0]
# large_h1_off = params_large_h1["Intercept"][0]
# large_h2_off = params_large_h2["Intercept"][0]
# large_h3_off = params_large_h3["Intercept"][0]
# large_h1_slope = -params_large_h1["Slope"][0]
# large_h2_slope = -params_large_h2["Slope"][0]
# large_h3_slope = -params_large_h3["Slope"][0]
#
# # Make fit
# ap_fit_small_h1 = gen_aperiodic(freq_I, (small_h1_off, small_h1_slope))
# ap_fit_med_h1 = gen_aperiodic(freq_I, (med_h1_off, med_h1_slope))
# ap_fit_med_h2 = gen_aperiodic(freq_I, (med_h2_off, med_h2_slope))
# ap_fit_large_h1 = gen_aperiodic(freq_I, (large_h1_off, large_h1_slope))
# ap_fit_large_h2 = gen_aperiodic(freq_I, (large_h2_off, large_h2_slope))
# ap_fit_large_h3 = gen_aperiodic(freq_I, (large_h3_off, large_h3_slope))
#
# plot_ap_fit_small_h1 = (freq_I, 10**ap_fit_small_h1, "b:")
# plot_ap_fit_med_h1 = (freq_I, 10**ap_fit_med_h1, "b:")
# plot_ap_fit_med_h2 = (freq_I, 10**ap_fit_med_h2/10, "b:")
# plot_ap_fit_large_h1 = (freq_I, 10**ap_fit_large_h1, "b:")
# plot_ap_fit_large_h2 = (freq_I, 10**ap_fit_large_h2/10, "b:")
# plot_ap_fit_large_h3 = (freq_I, 10**ap_fit_large_h3/10000, "b:")
# =============================================================================


# %% B Sim

band = (1, 599)

srate_sim = band[1] * 2 * h_max3  # avoid nyquist frequency
nperseg = srate_sim
welch_params = dict(fs=srate_sim, nperseg=nperseg)


# Oscillations parameters:
toy_slope = 2

periodic_params_s = [(freq_real_s, pow_real_s, bw_real_s/4)]
periodic_params_m = [(freq_real_m + 3, pow_real_m*5, bw_real_m/2.5)]
periodic_params_l = [(freq_real_l, pow_real_l*7, bw_real_l/2.3)]

# Sim Toy Signal
_, peak_small = osc_signals(toy_slope, periodic_params=periodic_params_s,
                            highpass=False, srate=srate_sim)
_, peak_med = osc_signals(toy_slope, periodic_params=periodic_params_m,
                          highpass=False, srate=srate_sim)
_, peak_large = osc_signals(toy_slope, periodic_params=periodic_params_l,
                            highpass=False, srate=srate_sim)

freq_b, peak_psd_small = sig.welch(peak_small, **welch_params)
freq_b, peak_psd_med = sig.welch(peak_med, **welch_params)
# freq_b, peak_psd_med12 = sig.welch(peak_med, **welch_params)
freq_b, peak_psd_large = sig.welch(peak_large, **welch_params)

# Filter 1-100Hz
filt_b = (freq_b <= 1000)
freq_b = freq_b[filt_b]
peak_psd_small = peak_psd_small[filt_b]
peak_psd_med = peak_psd_med[filt_b]
peak_psd_large = peak_psd_large[filt_b]

plot_psd_small = (freq_b, peak_psd_small, c_sim)
plot_psd_med = (freq_b, peak_psd_med, c_sim)
plot_psd_large = (freq_b, peak_psd_large, c_sim)

# plot_psd_small_low = (freq_b, peak_psd_small/10, c_sim)
plot_psd_med_low = (freq_b, peak_psd_med/100, c_sim)
plot_psd_large_low = (freq_b, peak_psd_large/100, c_sim)

# plot_psd_small_lower = (freq_b, peak_psd_small/100, c_sim)
# plot_psd_med_lower = (freq_b, peak_psd_med/100, c_sim)
plot_psd_large_lower = (freq_b, peak_psd_large/10000, c_sim)

# %% B Apply fooof to determine peak width

sim_s = FOOOF(max_n_peaks=1, verbose=False)
sim_m = FOOOF(max_n_peaks=1, peak_width_limits=(1, 150), verbose=False)
sim_l = FOOOF(max_n_peaks=1, peak_width_limits=(1, 150), verbose=False)

sim_s.fit(freq_b, peak_psd_small, (3, 20))
sim_m.fit(freq_b, peak_psd_med, (1, 200))
sim_l.fit(freq_b, peak_psd_large, (1, 150))

# sim_m.plot(plt_log=True)

bw_sim_s = sim_s.peak_params_[0, 2]
bw_sim_m = sim_m.peak_params_[0, 2]
bw_sim_l = sim_l.peak_params_[0, 2]

freq_sim_s = sim_s.peak_params_[0, 0]
freq_sim_m = sim_m.peak_params_[0, 0]
freq_sim_l = sim_l.peak_params_[0, 0]

# %% B Calc IRASA


# Change srate
irasa_params1["sf"] = srate_sim
irasa_params2["sf"] = srate_sim
irasa_params3["sf"] = srate_sim

irasa_params1["band"] = band
irasa_params2["band"] = band
irasa_params3["band"] = band

IRASA_sim_small_h1 = irasa(peak_small, **irasa_params1)

IRASA_sim_med_h1 = irasa(peak_med, **irasa_params1)
IRASA_sim_med_h2 = irasa(peak_med, **irasa_params2)
# IRASA_sim_med12_h1 = irasa(peak_med, **irasa_params1)
# IRASA_sim_med12_h2 = irasa(peak_med, **irasa_params2)

IRASA_sim_large_h1 = irasa(peak_large, **irasa_params1)
IRASA_sim_large_h2 = irasa(peak_large, **irasa_params2)
IRASA_sim_large_h3 = irasa(peak_large, **irasa_params3)

freqs_sim_s, ap_sim_small_h1, per1_s, params1_s = IRASA_sim_small_h1

freqs_sim_m, ap_sim_med_h1, per1_m, params1_m = IRASA_sim_med_h1
freqs_sim_m, ap_sim_med_h2, per2_m, params2_m = IRASA_sim_med_h2
# freqs_sim_m12, ap_sim_med12_h1, per1_m12, params1_m12 = IRASA_sim_med12_h1
# freqs_sim_m12, ap_sim_med12_h2, per2_m12, params2_m12 = IRASA_sim_med12_h2

freqs_sim_l, ap_sim_large_h1, per1_l, params1_l = IRASA_sim_large_h1
freqs_sim_l, ap_sim_large_h2, per2_l, params2_l = IRASA_sim_large_h2
freqs_sim_l, ap_sim_large_h3, per3_l, params3_l = IRASA_sim_large_h3

# normalize
ap_sim_small_h1 = ap_sim_small_h1[0]
ap_sim_med_h1 = ap_sim_med_h1[0]
ap_sim_med_h2 = ap_sim_med_h2[0]
# ap_sim_med12_h1 = ap_sim_med12_h1[0]
# ap_sim_med12_h2 = ap_sim_med12_h2[0]
ap_sim_large_h1 = ap_sim_large_h1[0]
ap_sim_large_h2 = ap_sim_large_h2[0]
ap_sim_large_h3 = ap_sim_large_h3[0]

plot_ap_sim_small_h1 = (freqs_sim_s, ap_sim_small_h1, c_IRASA1)

plot_ap_sim_med_h1 = (freqs_sim_m, ap_sim_med_h1, c_IRASA1)
plot_ap_sim_med_h2 = (freqs_sim_m, ap_sim_med_h2/100, c_IRASA2)
# plot_ap_sim_med12_h1 = (freqs_sim_m12, ap_sim_med12_h1, c_IRASA1)
# plot_ap_sim_med12_h2 = (freqs_sim_m12, ap_sim_med12_h2/10, c_IRASA2)

plot_ap_sim_large_h1 = (freqs_sim_l, ap_sim_large_h1, c_IRASA1)
plot_ap_sim_large_h2 = (freqs_sim_l, ap_sim_large_h2/100, c_IRASA2)
plot_ap_sim_large_h3 = (freqs_sim_l, ap_sim_large_h3/10000, c_IRASA3)

# =============================================================================
# # Extract fits
# sim_small_h1_off = params1_s["Intercept"][0]
# sim_small_h1_slope = -params1_s["Slope"][0]
# sim_med_h1_off = params1_s["Intercept"][0]
# sim_med_h2_off = params1_s["Intercept"][0]
# sim_med_h1_slope = -params1_s["Slope"][0]
# sim_med_h2_slope = -params1_s["Slope"][0]
# sim_large_h1_off = params1_s["Intercept"][0]
# sim_large_h2_off = params1_s["Intercept"][0]
# sim_large_h3_off = params1_s["Intercept"][0]
# sim_large_h1_slope = -params1_s["Slope"][0]
# sim_large_h2_slope = -params1_s["Slope"][0]
# sim_large_h3_slope = -params1_s["Slope"][0]
#
# # Make fit
# ap_fit_sim_small_h1 = gen_aperiodic(freqs_sim_s,
#                                     (sim_small_h1_off, sim_small_h1_slope))
# ap_fit_sim_med_h1 = gen_aperiodic(freqs_sim_s,
#                                   (sim_med_h1_off, sim_med_h1_slope))
# ap_fit_sim_med_h2 = gen_aperiodic(freqs_sim_s,
#                                   (sim_med_h2_off, sim_med_h2_slope))
# ap_fit_sim_large_h1 = gen_aperiodic(freqs_sim_s,
#                                     (sim_large_h1_off, sim_large_h1_slope))
# ap_fit_sim_large_h2 = gen_aperiodic(freqs_sim_s,
#                                     (sim_large_h2_off, sim_large_h2_slope))
# ap_fit_sim_large_h3 = gen_aperiodic(freqs_sim_s,
#                                     (sim_large_h3_off, sim_large_h3_slope))
#
# plot_ap_fit_sim_small_h1 = (freqs_sim_s, 10**ap_fit_sim_small_h1, "b:")
# plot_ap_fit_sim_med_h1 = (freqs_sim_s, 10**ap_fit_sim_med_h1, "b:")
# plot_ap_fit_sim_med_h2 = (freqs_sim_s, 10**ap_fit_sim_med_h2/10, "b:")
# plot_ap_fit_sim_large_h1 = (freqs_sim_s, 10**ap_fit_sim_large_h1, "b:")
# plot_ap_fit_sim_large_h2 = (freqs_sim_s, 10**ap_fit_sim_large_h2/10, "b:")
# plot_ap_fit_sim_large_h3 = (freqs_sim_s, 10**ap_fit_sim_large_h3/10000, "b:")
# =============================================================================

# %% Plot Params

width = 7.25  # inches
panel_fontsize = 12
legend_fontsize = 9
label_fontsize = 9
tick_fontsize = 9
annotation_fontsize = 7.5

mpl.rcParams['xtick.labelsize'] = tick_fontsize
mpl.rcParams['ytick.labelsize'] = tick_fontsize
mpl.rcParams['axes.labelsize'] = label_fontsize
mpl.rcParams['legend.fontsize'] = legend_fontsize
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


abc = dict(x=0, y=1.01, fontsize=panel_fontsize,
           fontdict=dict(fontweight="bold"))

# a11
ymini = -13
ymaxi = -7
yticks_a1 = 10**np.arange(ymini, ymaxi, dtype=float)
ylim_a1 = (yticks_a1[0], yticks_a1[-1])
yticklabels_a1 = [""] * len(yticks_a1)
yticklabels_a1[0] = fr"$10^{{{ymini}}}$"
yticklabels_a1[-1] = fr"$10^{{{ymaxi}}}$"
ylabel_a1 = "PSD [a.u.]"

xticklabels_a1 = []
xlim_a = (1, 100)
axes_a1 = dict(xticklabels=xticklabels_a1, xlim=xlim_a, yticks=yticks_a1,
               yticklabels=yticklabels_a1, ylim=ylim_a1)
freqs123 = [freq1, freq2, freq3]
colors123 = [c_range1, c_range2, c_range3]
text_dic = dict(x=100, ha="right", fontsize=annotation_fontsize)

# a12
xticks_a2 = [1, 10, 100]
yticks_a2 = [0, .5, 1]
xlabel_a2 = "Lower fitting range border [Hz]"
ylabel_a2 = r"$|a_{truth} - a_{IRASA}|$"
ylim_a2 = (0, 1)
axes_a2 = dict(xticks=xticks_a2, xticklabels=xticks_a2, yticks=yticks_a2,
               xlim=xlim_a, xlabel=xlabel_a2, ylim=ylim_a2, ylabel=ylabel_a2)

axes_a21 = dict(xticklabels=xticklabels_a1, xlim=xlim_a, yticks=yticks_a1,
                yticklabels=[], ylim=ylim_a1)
axes_a22 = dict(xticks=xticks_a2, xticklabels=xticks_a2, yticks=yticks_a2,
                yticklabels=[],
                xlim=xlim_a, xlabel=xlabel_a2, ylim=ylim_a2)

# d
ylabel_d = r"$\left(T/m\right)^2$/Hz"

# e
ylabel_e = r"$T^2/Hz$"

# f
ylabel_f = r"$\mu V^2/Hz$"

# b
xlim_b = (None, 700)
xlabel = "Frequency [Hz]"
xticks_b = [1, 10, 100, 1000]

# c
xticks_c = [1, 10, 100, 600]


def anno_fitrange(ax1, ax2, toy_psd, freqs, colors):
    vline_dic = dict(ls="--", clip_on=False, alpha=.3)
    for i, (freq_low, color) in enumerate(zip(freqs, colors)):
        ymin = ylim_a1[0]
        y = toy_psd[freq_low]
        xmin = freq_low
        xmax = upper_fitting_border
        coords = (y, xmin, xmax)
        ax1.hlines(*coords, color=color, ls="--")
        v_coords = (xmin, ymin, y)
        ax1.vlines(*v_coords, color=color, **vline_dic)

        # Add annotation
        s = f"{freq_low}-{xmax}Hz"
        if i == 0:
            s = "Fitting range: " + s
            y = y**.97
        else:
            y = y**.98
        ax1.text(s=s, y=y, **text_dic)
        # Add vlines below
        ymin = 0
        ymax = 1.4
        v_coords = (xmin, ymin, ymax)
        ax2.vlines(*v_coords, color=color, **vline_dic)


def annotate_fit_range(ax, xmin, xmax, height, ylow=None, yhigh=None,
                       annotate_middle=True, annotate_range=True):
    """
    Annotate fitting range.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Ax to draw the lines.
    xmin : float
        X-range minimum.
    xmax : float
        X-range maximum.
    ylow : float, optional
        Position on y-axis to connect the vertical lines. If None, no vertical
        lines are drawn. The default is None.
    yhigh : float, optional
        Position on y-axis to connect the vertical lines. If None, no vertical
        lines are drawn. The default is None.
    height : float
        Position on y-axis of range.
    annotate_middle : bool, optional
        Whether to annotate the frequency in the middle of the range (prettier)
        or if it is too small next to the range. The default is True.
    annotate_range : bool, optional
        Whether to annotate the range as min-max or the difference as max-min.
        By default, the range min-max is annotated.

    Returns
    -------
    None.

    """
    if annotate_middle:
        box_alpha = 1
        ha = "center"
        text_pos = 10**((np.log10(xmin) + np.log10(xmax)) / 2)
    else:
        box_alpha = 0
        ha = "right"
        text_pos = xmin * .9

    # Plot Values
    arrow_dic = dict(text="", xy=(xmin, height), xytext=(xmax, height),
                     arrowprops=dict(arrowstyle="|-|, widthA=.3, widthB=.3",
                                     shrinkA=0, shrinkB=0))
    anno_dic = dict(ha=ha, va="center", bbox=dict(fc="white", ec="none",
                    boxstyle="square,pad=0.2", alpha=box_alpha),
                    fontsize=annotation_fontsize)
    vline_dic = dict(color="k", lw=.5, ls=":")

    ax.annotate(**arrow_dic)
    if annotate_range:
        ax.text(text_pos, height, s=f"{xmin:.1f}-{xmax:.1f}Hz", **anno_dic)
    else:
        ax.text(text_pos, height, s=f"{xmax-xmin:.1f}Hz", **anno_dic)
    if ylow and yhigh:
        ax.vlines(xmin, height, ylow, **vline_dic)
        ax.vlines(xmax, height, yhigh, **vline_dic)


# %% Plot
fig = plt.figure(figsize=[width, 6.9], constrained_layout=True)

gs0 = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[10, 10, 1, 10])

gs00 = gs0[0].subgridspec(2, 3)
axA1 = fig.add_subplot(gs00[0, 0])
axA2 = fig.add_subplot(gs00[1, 0])
axB1 = fig.add_subplot(gs00[0, 1])
axB2 = fig.add_subplot(gs00[1, 1])
axC1 = fig.add_subplot(gs00[0, 2])
axC2 = fig.add_subplot(gs00[1, 2])

gs01 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[1])
ax5 = fig.add_subplot(gs01[0])
ax6 = fig.add_subplot(gs01[1])
ax7 = fig.add_subplot(gs01[2])

gs02 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[2])
ax_leg = fig.add_subplot(gs02[0])
ax_leg.axis("off")

gs03 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[3])
ax8 = fig.add_subplot(gs03[0])
ax9 = fig.add_subplot(gs03[1])
ax10 = fig.add_subplot(gs03[2])

# a
# a11
ax = axA1

# Plot sim
ax.loglog(freq_a, toy_psd_a, c_sim)
ax.loglog(freq0, psd_aperiodic_a, c_ap, zorder=0)
anno_fitrange(axA1, axA2, toy_psd_a, freqs123, colors123)

# Set axes
ax.text(s="a", **abc, transform=ax.transAxes)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.set(**axes_a1)
ax.set_ylabel(ylabel_a1, labelpad=-8)

# a12
ax = axA2
ax.semilogx(*error_plot_a)

# Set axes
ax.set(**axes_a2)

# a21
ax = axB1

# Plot sim
ax.loglog(freq_a, toy_psd_b, c_sim)
ax.loglog(freq0, psd_aperiodic_b, c_ap, zorder=0)
anno_fitrange(axB1, axB2, toy_psd_b, freqs123, colors123)

# Set axes
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.set(**axes_a21)

# a22
ax = axB2
ax.semilogx(*error_plot_b)

# Set axes
ax.set(**axes_a22)


# a31
ax = axC1
ax.loglog(freq_a, toy_psd_c, c_sim)
ax.loglog(freq0, psd_aperiodic_c, c_ap, zorder=0)
anno_fitrange(axC1, axC2, toy_psd_b, freqs123, colors123)

# Set axes
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ax.set(**axes_a21)

# a32
ax = axC2
ax.semilogx(*error_plot_c)

# Set axes
ax.set(**axes_a22)



# b1
ax = ax5
ax.loglog(*plot_psd_small)
ax.loglog(*plot_ap_sim_small_h1, label=r"$h_{max}$ = "f"{h_max1}")
# ax.loglog(*plot_ap_fit_sim_small_h1, label=f"a={sim_small_h1_slope:.2f}")

# ax.set_yticklabels([])
# xticks_bzz = ax.get_xticks()
# annotate freq bandwidth
xmin = freq_sim_s - bw_sim_s
xmax = freq_sim_s + bw_sim_s
ylow = plot_psd_small[1][np.argmin(np.abs(plot_psd_small[0] - xmin))]
yhigh = plot_psd_small[1][np.argmin(np.abs(plot_psd_small[0] - xmax))]
height = 1e-17
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=False, annotate_range=False)
ax.set_xticks(xticks_b)
ax.set_xticklabels(xticks_b)

# ax.legend(loc=1)
ax.set_ylabel(ylabel_a1)
ax.text(s="b", **abc, transform=ax.transAxes)
# ax.set_xlim(xlim_b)
# ax.set_ylim((1e-6, 1))

# b2
ax = ax6
ax.loglog(*plot_psd_med)
ax.loglog(*plot_ap_sim_med_h1, label=r"$h_{max}$ = "f"{h_max1}")
# ax.loglog(*plot_ap_fit_sim_med_h1, label=f"a={sim_med_h1_slope:.2f}")

ax.loglog(*plot_psd_med_low, alpha=.5)
ax.loglog(*plot_ap_sim_med_h2, label=r"$h_{max}$ = "f"{h_max2}")
# ax.loglog(*plot_ap_fit_sim_med_h2, label=f"a={sim_med_h2_slope:.2f}")
# ax.loglog(*plot_ap3_m, label=r"$h_{max}$ = "f"{h_max3}")
# ax.set_xlim(xlim_b)
# ax.set_ylim((1e-6, 1))
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([])
ax.set_yticklabels([], minor=True)

height = 1e-17
xmin = freq_sim_m - bw_sim_m
xmax = freq_sim_m + bw_sim_m
ylow = plot_psd_med_low[1][np.argmin(np.abs(plot_psd_med_low[0] - xmin))]
yhigh = plot_psd_med_low[1][np.argmin(np.abs(plot_psd_med_low[0] - xmax))]
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=False, annotate_range=False)
ax.set_xticks(xticks_b)
ax.set_xticklabels(xticks_b)

# ax.legend(fontsize=7)

# b3
ax = ax7
ax.loglog(*plot_psd_large)
ax.loglog(*plot_ap_sim_large_h1, label=r"$h_{max}$ = "f"{h_max1}")
# ax.loglog(*plot_ap_fit_sim_large_h1, label=f"a={sim_large_h1_slope:.2f}")

ax.loglog(*plot_psd_large_low, alpha=.5)
ax.loglog(*plot_ap_sim_large_h2, label=r"$h_{max}$ = "f"{h_max2}")
# ax.loglog(*plot_ap_fit_sim_large_h2, label=f"a={sim_large_h2_slope:.2f}")

ax.loglog(*plot_psd_large_lower, alpha=.5)
ax.loglog(*plot_ap_sim_large_h3, label=r"$h_{max}$ = "f"{h_max3}")
# ax.loglog(*plot_ap_fit_sim_large_h3, label=f"a={sim_large_h3_slope:.2f}")
# ax.set_xticks(xticks_b)
# ax.set_xticklabels(xticks_b)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([])
ax.set_yticklabels([], minor=True)


# annotate freq bandwidth
xmin = freq_sim_l - bw_sim_l# * 3
xmax = freq_sim_l + bw_sim_l# * 3
height = 1e-20
ylow = plot_psd_large_lower[1][np.argmin(np.abs(plot_psd_large_lower[0] - xmin))]
yhigh = plot_psd_large_lower[1][np.argmin(np.abs(plot_psd_large_lower[0] - xmax))]
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=False, annotate_range=False)
ax.set_xticks(xticks_b)
ax.set_xticklabels(xticks_b)

handles, labels = ax.get_legend_handles_labels()

ax_leg.legend(handles, labels, fontsize=legend_fontsize, ncol=3, loc=10)






alpha_long = .3


# c1
ax = ax8
ax.loglog(*plot_small)
ax.loglog(*plot_ap_small_h1, label=r"$h_{max}$ = "f"{h_max1}")
# ax.loglog(*plot_ap_fit_small_h1, label=f"a={small_h1_slope:.2f}")
ax.set_ylabel(ylabel_d)
ax.set_xlabel(xlabel)
# ax.set_xlim(xlim_b)
ymin, ymax = ax.get_ylim()
ax.set_ylim((3e-26, ymax))
ax.text(s="c", **abc, transform=ax.transAxes)
# annotate freq bandwidth
xmin = freq_real_s - bw_real_s
xmax = freq_real_s + bw_real_s
ylow = plot_small[1][np.argmin(np.abs(plot_small[0] - xmin))]
yhigh = plot_small[1][np.argmin(np.abs(plot_small[0] - xmax))]
height = 4e-26
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=False, annotate_range=False)
ax.set_xticks(xticks_c)
ax.set_xticklabels(xticks_c)


# c2
ax = ax9
ax.loglog(*plot_med)
ax.loglog(*plot_ap_med_h1, label=r"$h_{max}$ = "f"{h_max1}")
ax.loglog(*plot_ap_med_h1_long, ls="--", alpha=alpha_long, label=r"$h_{max}$ = "f"{h_max1}")
# ax.loglog(*plot_ap_fit_med_h1, label=f"a={med_h1_slope:.2f}")

ax.loglog(*plot_med_low, alpha=.5)
ax.loglog(*plot_ap_med_h2, label=r"$h_{max}$ = "f"{h_max2}")
ax.loglog(*plot_ap_med_h2_long, ls="--", alpha=alpha_long, label=r"$h_{max}$ = "f"{h_max2}")
# ax.loglog(*plot_ap_fit_med_h2, label=f"a={med_h2_slope:.2f}")
ax.set_ylabel(ylabel_e)
ax.set_xlabel(xlabel)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
# ax.set_xlim(xlim_b)
ax.set_ylim((3e-29, 2e-23))
# annotate freq bandwidth
xmin = freq_real_m - bw_real_m + 2
xmax = freq_real_m + bw_real_m + 2
ylow = plot_med_low[1][np.argmin(np.abs(plot_med_low[0] - xmin))]
yhigh = plot_med_low[1][np.argmin(np.abs(plot_med_low[0] - xmax))]
height = 1e-28
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=False, annotate_range=False)
ax.set_xticks(xticks_c)
ax.set_xticklabels(xticks_c)




# c3
ax = ax10
ax.loglog(*plot_large)
ax.loglog(*plot_ap_large_h1, label=r"$h_{max}$ = "f"{h_max1}")
ax.loglog(*plot_ap_large_h1_long, ls="--", alpha=alpha_long,
          label=r"$h_{max}$ = "f"{h_max1}")
# ax.loglog(*plot_ap_fit_large_h1, label=f"a={large_h1_slope:.2f}")

ax.loglog(*plot_large_low, alpha=.5)
ax.loglog(*plot_ap_large_h2, label=r"$h_{max}$ = "f"{h_max2}")
ax.loglog(*plot_ap_large_h2_long, ls="--", alpha=alpha_long,
          label=r"$h_{max}$ = "f"{h_max2}")
# ax.loglog(*plot_ap_fit_large_h2, label=f"a={large_h2_slope:.2f}")

ax.loglog(*plot_large_lower, alpha=.5)
ax.loglog(*plot_ap_large_h3, label=r"$h_{max}$ = "f"{h_max3}")
ax.loglog(*plot_ap_large_h3_long, ls="--", alpha=alpha_long,
          label=r"$h_{max}$ = "f"{h_max3}")
# ax.loglog(*plot_ap_fit_large_h3, label=f"a={large_h3_slope:.2f}")
ax.set_ylabel(ylabel_f)
ax.set_xlabel(xlabel)
# ax.set_xlim(xlim_b)
ax.set_ylim((1e-9, 2))
x_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.xaxis.set_minor_locator(x_minor)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)

# annotate freq bandwidth
xmin = freq_real_l - bw_real_l
xmax = freq_real_l + bw_real_l
ylow = plot_large_lower[1][np.argmin(np.abs(plot_large_lower[0] - xmin))]
yhigh = plot_large_lower[1][np.argmin(np.abs(plot_large_lower[0] - xmax))]
height = 2e-8
annotate_fit_range(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                   height=height, annotate_middle=False, annotate_range=False)
ax.set_xticks(xticks_b)
ax.set_xticklabels(xticks_b)
# ax.legend(fontsize=7)

plt.savefig(fig_path + fig_name, bbox_inches="tight")
plt.show()




"""
Message:

Very broad peak widths require very large resampling factors.

C2 and C3: Show that small H does not capture true aperiodic component.
But how to choose H?
In one case, h_max is chosen to avoid highpass and noise floor (short)
in the other h_max is chosen to identify peaks (shorter).


Discuss problems:

Problem 1: I want to show that small h-values don't work for real spectra if
the peak widths are too broad. However, the message "dissappears" because real
spectra have other strong difficulties such as highpass and noise plateau.
This makes my message difficult to deliver, it becomes ambigious because of
the other difficulties which are topic of the last figure.

Problem 2: How to choose H values? Noise plateua is ambigous.

Which h values to choose for real data?

Problem 1:
-> when h becomes too small/peak too broad irasa gets worse gradually.
Difficult to choose which h values to show in plot as example for
(un)successful fits.

Problem 2:
If I want to show maximum possible h value (as Gabriel suggested) lower border
is clear but plateau is less well defined. Plateau might be not flat but a
little flatter than beginning.

Problem 2: Irasa does not behave exactly as I expect. For medium peak width
and large peak width intermediate h value works just as fine.
It should work for medium and not work for large. Is the reason maybe in the
Center frequency of the two peaks 12Hz vs 40Hz?
-> test in simulation by shifting center freq! If yes: difficult to show
effect in real data. If no: is it shape of the peak? What is it?

Problem 3: It depends on the center frequency!!!

(Color orangered is NOT ok. Too similar to orange, especially with low alpha.)
"""

"""
To do:
    A:
        - annotate peak widths
        - if all peaks similar width: increase peak widths and y-error axis
    
    C:
        - Ask Group: LFP+Mag+Grad? Only LFP + MEG?
"""











