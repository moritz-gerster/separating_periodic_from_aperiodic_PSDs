"""Explain fooof and IRASA."""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.fft import irfft, rfftfreq
import scipy as sp
import scipy.signal as sig
from pathlib import Path
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from fooof.plts.spectra import plot_spectrum
# from fooof.plts.annotate import plot_annotated_peak_search
from fooof.plts.annotate import plot_annotated_peak_search_MG
import matplotlib.gridspec as gridspec
from noise_helper import irasa
import fractions
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


# =============================================================================
# def calc_error(signal):
#     """Fit IRASA and subtract ground truth to obtain fitting error."""
#     fit_errors = []
#     for i in trange(len(lower_fitting_borders)):
#         freq_range = (lower_fitting_borders[i], upper_fitting_border)
#         _, _, _, params = irasa(data=signal, band=freq_range, sf=srate)
#         exp = -params["Slope"][0]
#         error = np.abs(toy_slope - exp)
#         fit_errors.append(error)
#     return fit_errors
# =============================================================================


# %% PARAMETERS

# Signal params
srate = 2400
welch_params_f = dict(fs=srate, nperseg=1*srate)
welch_params_I = dict(fs=srate, nperseg=4*srate)

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig0_Intro.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)


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
# c_IRASA3 = "C4"

lw = 2


# %% Make signal
toy_slope = 1
freq1 = 10  # Hz
freq2 = 25  # Hz
amp1 = .02
amp2 = .01
width = .001

periodic_params = [(freq1, amp1, width),
                   (freq2, amp2, width)]

# Sim Toy Signal
toy_aperiodic, toy_comb = osc_signals(toy_slope,
                                        periodic_params=periodic_params,
                                        highpass=False)

_, toy_osc = osc_signals(0,
                         periodic_params=periodic_params,
                         highpass=False)

freqs_f, psd_comb_f = sig.welch(toy_comb, **welch_params_f)
_, psd_osc_f = sig.welch(toy_osc, **welch_params_f)
freqs_I, psd_comb_I = sig.welch(toy_comb, **welch_params_I)
_, psd_osc_I = sig.welch(toy_osc, **welch_params_I)

# Filter 1-100Hz
# mask = (freqs <= 100)
# freqs = freqs[mask]
# psd_comb = psd_comb[mask]
# psd_osc = psd_osc[mask]

# %% Calc fooof

# Set whether to plot in log-log space
plt_log = True
freq_range = (1, 100)

fm = FOOOF() # use default settings
fm.add_data(freqs_f, psd_comb_f, freq_range)

# Fit the power spectrum model
fm.fit(freqs_f, psd_comb_f, freq_range)

# Do an initial aperiodic fit - a robust fit, that excludes outliers
init_ap_fit = gen_aperiodic(fm.freqs,
                            fm._robust_ap_fit(fm.freqs, fm.power_spectrum))

# Recompute the flattened spectrum using the initial aperiodic fit
init_flat_spec = fm.power_spectrum - init_ap_fit


# %% Combine fooof

fig_width = 7.25

fig, axes = plt.subplots(3, 3, figsize=[fig_width*3, 6*3])

# plot initial fit
ax = axes[0, 0]
plot_spectrum(fm.freqs, fm.power_spectrum, plt_log,
              label='Original Power Spectrum', color='black', ax=ax)
plot_spectrum(fm.freqs, init_ap_fit, plt_log, label='Initial Aperiodic Fit',
              color='blue', alpha=0.5, linestyle='dashed', ax=ax)


# Plot the flattened the power spectrum
ax = axes[0, 1]
plot_spectrum(fm.freqs, init_flat_spec, plt_log,
              label='Flattened Spectrum', color='black', ax=ax)

# show iterations
ax = axes[0, 2]
plot_annotated_peak_search_MG(fm, 0, ax)

ax = axes[1, 0]
plot_annotated_peak_search_MG(fm, 1, ax)

ax = axes[1, 1]
plot_annotated_peak_search_MG(fm, 2, ax)


# plot model
ax = axes[2, 1]
plot_spectrum(fm.freqs, fm._peak_fit, plt_log, color='green',
              label='Final Periodic Fit', ax=ax)

# plot aperiodic
ax = axes[2, 2]
plot_spectrum(fm.freqs, fm._spectrum_peak_rm, plt_log,
              label='Peak Removed Spectrum', color='black', ax=ax)
plot_spectrum(fm.freqs, fm._ap_fit, plt_log, label='Final Aperiodic Fit',
              color='blue', alpha=0.5, linestyle='dashed', ax=ax)

# including real data
# fm.plot(plt_log=plt_log)

# %% Calc IRASA
hset = np.arange(1.1, 1.95, 0.05)

irasa_params = dict(sf=srate, band=freq_range, win_sec=4, hset=hset)

IRASA = irasa(data=toy_comb, **irasa_params)

freqs_irasa, psd_ap, psd_osc, params = IRASA

psd_ap, psd_osc = psd_ap[0], psd_osc[0]

plt.loglog(freqs_irasa, psd_ap)
plt.loglog(freqs_irasa, psd_osc)

# %% 1

hset = [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
          1.65, 1.7, 1.75, 1.8, 1.85, 1.9]
hset = [1.1, 1.5, 2]

win = welch_params_I["nperseg"]
psds_resampled = np.zeros((len(hset), *psd_comb_I.shape))

for i, h in enumerate(hset):
    # Get the upsampling/downsampling (h, 1/h) factors as integer
    rat = fractions.Fraction(str(h))
    up, down = rat.numerator, rat.denominator
    # Much faster than FFT-based resampling
    data_up = sig.resample_poly(toy_comb, up, down)
    data_down = sig.resample_poly(toy_comb, down, up, axis=-1)

    freqs_up, psd_up = sig.welch(data_up, h * srate, nperseg=win)
    freqs_dw, psd_dw = sig.welch(data_down, srate / h, nperseg=win)
    
    psds_resampled[i, :] = np.sqrt(psd_up * psd_dw)

    
for i, h in enumerate(hset):
    plt.loglog(psds_resampled[i])

# Now we take the median PSD of all the resampling factors, which gives
# a good estimate of the aperiodic component of the PSD.
psd_aperiodic = np.median(psds_resampled, axis=0)

plt.loglog(psd_aperiodic)

# We can now calculate the oscillations (= periodic) component.
psd_osc = psd_comb_I - psd_aperiodic

IR_offset = params["Intercept"][0]
IR_slope = -params["Slope"][0]

psd_fit = gen_aperiodic(freqs_irasa, (IR_offset, IR_slope))

# %% Combine IRASA
mask = freqs_I < 100

fig, axes = plt.subplots(2, 2,  figsize=[fig_width*3, 6*3], sharex=False)

ax = axes[0, 0]
ax.loglog(freqs_I[mask], psd_comb_I[mask])

ax = axes[0, 1]
ax.loglog(freqs_I[mask], psd_comb_I[mask], "k")
for i in range(len(hset)):
    ax.loglog(freqs_I[mask], psds_resampled[i, mask])

ax = axes[1, 0]
ax.loglog(freqs_I[mask], psd_aperiodic[mask])

ax = axes[1, 1]
ax.semilogx(freqs_I[mask], psd_osc[mask])

    # %% Compare fooof IRASA


fig = plt.figure(figsize=[fig_width*3, 5.5*3], constrained_layout=True)

gs0 = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1])

gs00 = gs0[0].subgridspec(1, 5, width_ratios=[5, 5, 2, 2, 2])

ax1 = fig.add_subplot(gs00[0])
ax2 = fig.add_subplot(gs00[1])
ax3 = fig.add_subplot(gs00[2])
ax4 = fig.add_subplot(gs00[3])
ax5 = fig.add_subplot(gs00[4])

gs01 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[1])
ax6 = fig.add_subplot(gs01[0])
ax7 = fig.add_subplot(gs01[1])
ax8 = fig.add_subplot(gs01[2])

gs02 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[2])
ax9 = fig.add_subplot(gs02[0])
#ax7 = fig.add_subplot(gs01[1])
#ax8 = fig.add_subplot(gs01[2])



ax = ax1
plot_spectrum(fm.freqs, fm.power_spectrum, plt_log,
              label='Original Power Spectrum', color='black', ax=ax)
plot_spectrum(fm.freqs, init_ap_fit, plt_log, label='Initial Aperiodic Fit',
              color='blue', alpha=0.5, linestyle='dashed', ax=ax)


# Plot the flattened the power spectrum
ax = ax2
plot_spectrum(fm.freqs, init_flat_spec, plt_log,
              label='Flattened Spectrum', color='black', ax=ax)

# show iterations
ax = ax3
plot_annotated_peak_search_MG(fm, 0, ax)
ax.get_legend().remove()

ax = ax4
plot_annotated_peak_search_MG(fm, 1, ax)
ax.get_legend().remove()
ax.set_ylabel("")
ax.set_yticks([])
ax.set_yticklabels([])

ax = ax5
plot_annotated_peak_search_MG(fm, 2, ax)
ax.set_ylabel("")
ax.set_yticks([])
ax.set_yticklabels([])



# IRASA
ax = ax6
ax.loglog(freqs_I, psd_comb_I)

ax = ax7
ax.loglog(freqs_I, psd_comb_I, "k")
for i in range(len(hset)):
    ax.loglog(freqs_I, psds_resampled[i])
    
ax = ax8
ax.loglog(freqs_I, psd_aperiodic, label="IRASA")
ax.loglog(freqs_irasa, 10**psd_fit, label="IRASA fit")
ax.loglog(fm.freqs, 10**fm._spectrum_peak_rm, label="fooof")
ax.loglog(fm.freqs, 10**fm._ap_fit, label="fooof fit")
ax.legend()

ax = ax9
ax.plot(fm.freqs, 10**fm._peak_fit, label="osc fooof")
osc_irasa *= 1e12
osc_irasa = np.log10(osc_irasa)
osc_irasa = np.nan_to_num(osc_irasa)
ax.plot(freqs_I, osc_irasa, label="osc IRASA")
ax.set_xlim(1, 100)
ax.legend()

plt.show()
# %%






# Plot full model, created by combining the peak and aperiodic fits
plot_spectrum(fm.freqs, fm.fooofed_spectrum_, plt_log,
              label='Full Model', color='red')
