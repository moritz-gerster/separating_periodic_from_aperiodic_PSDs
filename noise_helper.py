"""Helper functions."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from scipy.stats import norm
import scipy.signal as sig
from fooof import FOOOF, FOOOFGroup
import matplotlib.ticker
import fractions
from pathlib import Path


def noise_white(samples, seed=True):
    """Create White Noise of N samples."""
    if seed:
        np.random.seed(10)
    noise = np.random.normal(0, 1, size=samples)
    return noise


def osc_signals(samples, slope, freq_osc, amp, width=None, seed=True,
                srate=2400, normalize=True):
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
        if normalize:
            # normalize
            return (noise - noise.mean()) / noise.std()
        else:
            return noise
    elif isinstance(slope, (np.ndarray, list)):
        pink_noises = np.zeros([len(slope), samples-2])
        for i in range(len(slope)):
            # Multiply Amp Spectrum by 1/f
            amps_i = amps / freqs ** (slope[i] / 2)  # half slope needed
            amps_i *= np.exp(1j * random_phases)
            # Transform back to get pink noise time series
            noise = np.fft.irfft(amps_i)
            if normalize:
                # normalize
                pink_noises[i] = (noise - noise.mean()) / noise.std()
            else:
                pink_noises[i] = noise
        return pink_noises


def osc_signals_correct(samples, slopes, freq_osc=[], amp=[], width=[],
                        seed=False,
                        srate=2400, normalize=False):
    """
    Generate a mixture of 1/f-aperiodic and periodic signals.

    EDIT: First make 1/f, THEN add peaks.

    Parameters
    ----------
    samples : int
        Number of signal samples.
    freq_osc : list of floats
        Peak frequencies.
    amp : list of floats
        Amplitudes in relation to noise.
    slope : list of floats
        1/f-slope values.
    width : list of floats
        Standard deviation of Gaussian peaks. The default is None which
        corresponds to sharp delta peaks.

    Returns
    -------
    pink_noise : ndarray of size slope.size X samples
        Return signal.
    """
    if seed:
        np.random.seed(seed)
    # Initialize output
    noises = np.zeros([len(slopes), samples-2])
    # Make fourier amplitudes
    amps = np.ones(samples//2 + 1, complex)
    freqs = np.fft.rfftfreq(samples, d=1/srate)

    # Make 1/f
    amps, freqs, = amps[1:], freqs[1:]  # avoid divison by 0
    # Generate random phases
    random_phases = np.random.uniform(0, 2*np.pi, size=amps.shape)
    print(seed)
    print(random_phases[0])
    

    for j, slope in enumerate(slopes):
        # Multiply Amp Spectrum by 1/f
        amps = amps / freqs ** (slope / 2)  # half slope needed: 1/f^2 in power spectrum = sqrt(1/f^2)=1/f^2*0.5=1/f in amp spectrum
        amps *= np.exp(1j * random_phases)

        for i in range(len(freq_osc)):
            freq_idx = np.abs(freqs - freq_osc[i]).argmin()
            # make Gaussian peak
            amp_dist = norm(freq_osc[i], width[i]).pdf(freqs)
            # amp_dist /= np.max(amp_dist)    # check ich nciht
            amps += amp[i] * amp_dist

        noises[j] = np.fft.irfft(amps)
    return noises









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


def slope_error(slopes, freq, noise_psds, freq_range, IRASA,
                fooof_params=None):
    """
    Calculate fooof and IRASA slope estimation difference to ground truth.

    Fooof needs to be calculated to obtain an array of all estimates,
    RASA need to be calculated before.

    Parameters
    ----------
    slopes : ndarray
        Ground truth.
    freq : ndarray
        Frequencies.
    noise_psds : ndarray
        PSDs.
    freq_range : tuple of floats
        Freq range used for estimation.
    IRASA : tuple
        IRASA output. Only IRASA fit params are needed.

    Returns
    -------
    ndarray
        Ground truth - fooof estimates.
    ndarray
        Ground truth - IRASA estimates.

    """
    if fooof_params:
        fg = FOOOFGroup(**fooof_params)  # Init fooof
    else:
        fg = FOOOFGroup()  # Init fooof
    fg.fit(freq, noise_psds, freq_range)
    slopes_f = fg.get_params("aperiodic", "exponent")
    if IRASA:
        _, _, _, slopes_i = IRASA
        slopes_i = slopes_i["Slope"].to_numpy()
        slopes_i = -slopes_i  # make fooof and IRASA slopes comparable
        return slopes-slopes_f, slopes-slopes_i
    else:
        return slopes-slopes_f, None

def plot_all(freq, noise_psds, freq_f, noise_psds_f, IRASA, slopes, freq_range,
             white_ratio, plot_osc=False, save_path=None, save_name=None,
             add_title=None, win_sec=None):
    """
    Plot original spectrum + fooof and IRASA fits and error.

    Parameters
    ----------
    freq : ndarray
        Frequency bins of psd.
    noise_psds : ndarray
        Array of noises with different 1/f slopes..
    IRASA : tuple
        Result of yasa.irasa function.
    slopes : ndarray
        1/f slopes.
    freq_range : tuple of floats
        Fitting range for both methods.
    white_ratio : float
        Amount of white noise. 1 corresponds to 100% (SNR 50/50).
    plot_osc : boolean, optional
        Whether to plot oscillatory peaks for IRASA. The default is False.
    save_path : str, optional
        Save path. The default is None.
    save_name : str, optional
        Save name. The default is None.
    add_title : str, optional
        Title to add for description. The default is None.

    Returns
    -------
    Fig.
    """
    fig, axes = plt.subplots(1, 4, figsize=[17, 6])

    ax = axes[0]
    mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
    for i in range(slopes.size):
        ax.loglog(freq[mask], noise_psds[i, mask],
                  label=f"1/f={slopes[i]:.2f}")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ylim = ax.get_ylim()
    xticks = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 4)
    ax.set_xticks(xticks)
    xticklabels = [np.round(xtick) for xtick in xticks]
    ax.set_xticklabels(xticklabels)
    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()
    ax.set_ylabel("Power")
    ax.set_xlabel('Frequency')
    ax.set_title("1/f Noise: Ground truth")
    ax.legend(title="Ground truth", loc=3, ncol=2,
              bbox_transform=fig.transFigure,
              bbox_to_anchor=[0.12, -.285])

    fooof_params = dict(max_n_peaks=0, verbose=False)  # no oscillations
    err_f, err_i = slope_error(slopes, freq_f, noise_psds_f, freq_range, IRASA,
                               fooof_params=fooof_params)
    err_sum_f = np.sum(np.abs(err_f))

    ax = axes[1]
    labels = []
    for i, noise_psd in enumerate(noise_psds_f):
        print(f"...fitting fooof {i+1} of {len(slopes)}")
        # fm = FOOOF(**fooof_params)  # Init fooof
        fm = FOOOF()  # Init fooof
        # try:
        fm.fit(freq_f, noise_psd, freq_range)
        fm.plot(plt_log=True, ax=ax)
        exponent = fm.get_params('aperiodic_params', 'exponent')
        labels.append(f" 1/f={exponent:.2f}")
        # except:
        # labels.append(" 1/f=failed")
    handles, _ = ax.get_legend_handles_labels()
    handles = handles[2::3] + [handles[-2]]
    labels = labels + ["osc"]
    ax.legend(handles, labels, title="fooof", loc=3, ncol=2,
              bbox_transform=fig.transFigure,
              bbox_to_anchor=[0.322, -.285])
    freq_name = f"{freq_range[0]}-{freq_range[1]}Hz"
    ax.set_title(f"Fooof error: {err_sum_f:.2f}")
    ax.set_ylabel("")
    ax.set_xlabel('Frequency', fontsize='x-large')
    ax.set_yticks(np.log10(yticks))
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(np.log10(xticks))
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(np.log10(ylim))
    ax.grid(False)

    ax = axes[2]
    if IRASA:
        freq_i, aperiodic, osc, params = IRASA
        # normalize
        # if normalize:
        #    aperiodic /= aperiodic.max(1)[:, np.newaxis]
        for i in range(slopes.size):
            ax.loglog(freq_i, aperiodic[i], "b", linestyle="--", lw=2,
                      label="aperiodic")
            if plot_osc:
                ax.loglog(freq_i, osc[i] + aperiodic[i], "r", lw=2,
                          label="osc", alpha=.5)
        handles, _ = ax.get_legend_handles_labels()
        labels = [f"1/f={-params['Slope'].iloc[i]:.2f}"
                  # f"     Offset={params['Intercept'].iloc[i]:.2f}"
                  for i in range(slopes.size)]
        if plot_osc:
            labels = labels + ["osc"]
            handles = handles[::2] + [handles[-1]]
        else:
            pass
        ax.legend(handles, labels, title="IRASA", loc=3, ncol=2,
                  bbox_transform=fig.transFigure,
                  bbox_to_anchor=[0.525, -.285])
        # ax.set_ylim([ymin, ymax])
        # ax.set_xlim([1, 600])
        err_sum_i = np.sum(np.abs(err_i))
        ax.set_title(f"IRASA error: {err_sum_i:.2f}")
        ax.set_xlabel('Frequency')
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        # ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    ax = axes[3]
    ax.plot(slopes, slopes-slopes, "k", label="Ground Truth")
    ax.plot(slopes, err_f, "r", label="fooof")
    if IRASA:
        ax.plot(slopes, err_i, "b", label="IRASA")
    ax.legend()
    ax.set_xticks(slopes[::2])
    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()
    ax.secondary_yaxis('right')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel("1/f")
    # ax.set_ylabel("Fitting error")
    ax.set_title("Ground truth - fitting value")
    suptitle = (f"Fit Range: {freq_name}, White noise: {white_ratio*100}%, "
                f"Welch window: {win_sec}s")
    if add_title:
        suptitle += ", " + add_title
    plt.suptitle(suptitle, position=[0.5, 1.02], fontsize=20)

    if save_name:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + save_name, bbox_inches="tight")
    plt.show()


def calc_psd(x, fs=1.0, nperseg=None, axis=-1, average='mean', **kwargs):
    """Kommt noch."""
    if average == 'mean':
        def average(x):
            return np.nanmean(x, axis=-1)
    if average == 'median':
        def average(x):
           return (np.nanmedian(x, axis=-1) /
                   sig.spectral._median_bias(x.shape[-1]))
    f, t, csd = sig.spectral._spectral_helper(x, x,
            fs=fs, nperseg=nperseg, axis=-1, mode='psd', **kwargs)
    # calculate the requested average
    try:
        csd_mean = average(csd)
    except(TypeError):
        'average must be a function, got %' % type(average)
    else:
        return f, csd_mean


def irasa(data, sf=None, ch_names=None, band=(1, 30),
          hset=[1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
          1.65, 1.7, 1.75, 1.8, 1.85, 1.9], return_fit=True, win_sec=4,
          reject_bad_segs=True,
          kwargs_welch=dict(average='mean', window='hann')):
    r"""
    TO DO: REJECT BAD SEGMENTS: TRUE
     if mne reject bad data and set nan 
     
     Fit ist 1e6 zu groß
    
    
    Separate the aperiodic (= fractal, or 1/f) and oscillatory component
    of the power spectra of EEG data using the IRASA method.

    .. versionadded:: 0.1.7

    Parameters
    ----------
    data : :py:class:`numpy.ndarray` or :py:class:`mne.io.BaseRaw`
        1D or 2D EEG data. Can also be a :py:class:`mne.io.BaseRaw`, in which
        case ``data``, ``sf``, and ``ch_names`` will be automatically
        extracted, and ``data`` will also be converted from Volts (MNE default)
        to micro-Volts (YASA).
    sf : float
        The sampling frequency of data AND the hypnogram.
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...]. If None,
        channels will be labelled ['CHAN000', 'CHAN001', ...].
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    band : tuple or None
        Broad band frequency range.
        Default is 1 to 30 Hz.
    hset : list or :py:class:`numpy.ndarray`
        Resampling factors used in IRASA calculation. Default is to use a range
        of values from 1.1 to 1.9 with an increment of 0.05.
    return_fit : boolean
        If True (default), fit an exponential function to the aperiodic PSD
        and return the fit parameters (intercept, slope) and :math:`R^2` of
        the fit.

        The aperiodic signal, :math:`L`, is modeled using an exponential
        function in semilog-power space (linear frequencies and log PSD) as:

        .. math:: L = a + \text{log}(F^b)

        where :math:`a` is the intercept, :math:`b` is the slope, and
        :math:`F` the vector of input frequencies.
    win_sec : int or float
        The length of the sliding window, in seconds, used for the Welch PSD
        calculation. Ideally, this should be at least two times the inverse of
        the lower frequency of interest (e.g. for a lower frequency of interest
        of 0.5 Hz, the window length should be at least 2 * 1 / 0.5 =
        4 seconds).
    kwargs_welch : dict
        Optional keywords arguments that are passed to the
        :py:func:`scipy.signal.welch` function.

    Returns
    -------
    freqs : :py:class:`numpy.ndarray`
        Frequency vector.
    psd_aperiodic : :py:class:`numpy.ndarray`
        The fractal (= aperiodic) component of the PSD.
    psd_oscillatory : :py:class:`numpy.ndarray`
        The oscillatory (= periodic) component of the PSD.
    fit_params : :py:class:`pandas.DataFrame` (optional)
        Dataframe of fit parameters. Only if ``return_fit=True``.

    Notes
    -----
    The Irregular-Resampling Auto-Spectral Analysis (IRASA) method is
    described in Wen & Liu (2016). In a nutshell, the goal is to separate the
    fractal and oscillatory components in the power spectrum of EEG signals.

    The steps are:

    1. Compute the original power spectral density (PSD) using Welch's method.
    2. Resample the EEG data by multiple non-integer factors and their
       reciprocals (:math:`h` and :math:`1/h`).
    3. For every pair of resampled signals, calculate the PSD and take the
       geometric mean of both. In the resulting PSD, the power associated with
       the oscillatory component is redistributed away from its original
       (fundamental and harmonic) frequencies by a frequency offset that varies
       with the resampling factor, whereas the power solely attributed to the
       fractal component remains the same power-law statistical distribution
       independent of the resampling factor.
    4. It follows that taking the median of the PSD of the variously
       resampled signals can extract the power spectrum of the fractal
       component, and the difference between the original power spectrum and
       the extracted fractal spectrum offers an approximate estimate of the
       power spectrum of the oscillatory component.

    Note that an estimate of the original PSD can be calculated by simply
    adding ``psd = psd_aperiodic + psd_oscillatory``.

    For an example of how to use this function, please refer to
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/09_IRASA.ipynb

    References
    ----------
    [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
    Components in the Power Spectrum of Neurophysiological Signal.
    Brain Topography, 29(1), 13–26.
    https://doi.org/10.1007/s10548-015-0448-0

    [2] https://github.com/fieldtrip/fieldtrip/blob/master/specest/

    [3] https://github.com/fooof-tools/fooof

    [4] https://www.biorxiv.org/content/10.1101/299859v1
    """
    # Check if input data is a MNE Raw object
    if isinstance(data, mne.io.BaseRaw):
        sf = data.info['sfreq']  # Extract sampling frequency
        ch_names = data.ch_names  # Extract channel names
        # Convert from V to uV
        data = data.get_data(reject_by_annotation="nan") * 1e6
    else:
        # Safety checks
        assert isinstance(data, np.ndarray), 'Data must be a numpy array.'
        data = np.atleast_2d(data)
        assert data.ndim == 2, 'Data must be of shape (nchan, n_samples).'
        nchan, npts = data.shape
        assert nchan < npts, 'Data must be of shape (nchan, n_samples).'
        assert sf is not None, 'sf must be specified if passing a numpy array.'
        assert isinstance(sf, (int, float))
        if ch_names is None:
            ch_names = ['CHAN' + str(i).zfill(3) for i in range(nchan)]
        else:
            ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
            assert ch_names.ndim == 1, 'ch_names must be 1D.'
            assert len(ch_names) == nchan, 'ch_names must match data.shape[0].'

    # Check the other arguments
    hset = np.asarray(hset)
    assert hset.ndim == 1, 'hset must be 1D.'
    assert hset.size > 1, '2 or more resampling fators are required.'
    hset = np.round(hset, 4)  # avoid float precision error with np.arange.
    band = sorted(band)
    assert band[0] > 0, 'first element of band must be > 0.'
    assert band[1] < (sf / 4), 'second element of band should be < (sf / 4).'
    win = int(win_sec * sf)  # nperseg

    # Calculate the original PSD over the whole data
# =============================================================================
#     CHANGED TO ALLOW NAN VALUES
    freqs, psd = calc_psd(data, sf, nperseg=win, **kwargs_welch)
# =============================================================================

    # Start the IRASA procedure
    psds = np.zeros((len(hset), *psd.shape))

    for i, h in enumerate(hset):
        # Get the upsampling/downsampling (h, 1/h) factors as integer
        rat = fractions.Fraction(str(h))
        up, down = rat.numerator, rat.denominator
        # Much faster than FFT-based resampling
        data_up = sig.resample_poly(data, up, down, axis=-1)
        data_down = sig.resample_poly(data, down, up, axis=-1)
        # Calculate the PSD using same params as original
# =============================================================================
#     CHANGED TO ALLOW NAN VALUES
        freqs_up, psd_up = calc_psd(data_up, h * sf, nperseg=win,
                                        **kwargs_welch)
        freqs_dw, psd_dw = calc_psd(data_down, sf / h, nperseg=win,
                                        **kwargs_welch)
# =============================================================================
        # Geometric mean of h and 1/h
        psds[i, :] = np.sqrt(psd_up * psd_dw)

    # Now we take the median PSD of all the resampling factors, which gives
    # a good estimate of the aperiodic component of the PSD.
    psd_aperiodic = np.median(psds, axis=0)

    # We can now calculate the oscillations (= periodic) component.
    psd_osc = psd - psd_aperiodic

    # Let's crop to the frequencies defined in band
    mask_freqs = np.ma.masked_outside(freqs, *band).mask
    freqs = freqs[~mask_freqs]
    psd_aperiodic = np.compress(~mask_freqs, psd_aperiodic, axis=-1)
    psd_osc = np.compress(~mask_freqs, psd_osc, axis=-1)

    if return_fit:
        # Aperiodic fit in semilog space for each channel
        from scipy.optimize import curve_fit
        intercepts, slopes, r_squared = [], [], []

        def func(t, a, b):
            # See https://github.com/fooof-tools/fooof
# =============================================================================
#             CORRECTED ERROR: NP.LOG -> NP.LOG10
            return a + np.log10(t**b)
# =============================================================================

        for y in np.atleast_2d(psd_aperiodic):
# =============================================================================
#             CORRECTED ERROR: NP.LOG -> NP.LOG10
            y_log = np.log10(y)
# =============================================================================
            # Note that here we define bounds for the slope but not for the
            # intercept.
            popt, pcov = curve_fit(func, freqs, y_log, p0=(2, -1),
                                   bounds=((-np.inf, -10), (np.inf, 2)))
            intercepts.append(popt[0])
            slopes.append(popt[1])
            # Calculate R^2: https://stackoverflow.com/q/19189362/10581531
            residuals = y_log - func(freqs, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_log - np.mean(y_log))**2)
            r_squared.append(1 - (ss_res / ss_tot))

        # Create fit parameters dataframe
        fit_params = {'Chan': ch_names, 'Intercept': intercepts,
                      'Slope': slopes, 'R^2': r_squared,
                      'std(osc)': np.std(psd_osc, axis=-1, ddof=1)}
        return freqs, psd_aperiodic, psd_osc, pd.DataFrame(fit_params)
    else:
        return freqs, psd_aperiodic, psd_osc
