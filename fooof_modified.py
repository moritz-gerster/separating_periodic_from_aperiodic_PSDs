#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:04:12 2021

@author: moritzgerster
"""
"""FOOOF Object - base object which defines the model.

Private Attributes
==================
Private attributes of the FOOOF object are documented here.

Data Attributes
---------------
_spectrum_flat : 1d array
    Flattened power spectrum, with the aperiodic component removed.
_spectrum_peak_rm : 1d array
    Power spectrum, with peaks removed.

Model Component Attributes
--------------------------
_ap_fit : 1d array
    Values of the isolated aperiodic fit.
_peak_fit : 1d array
    Values of the isolated peak fit.

Internal Settings Attributes
----------------------------
_ap_percentile_thresh : float
    Percentile threshold for finding peaks above the aperiodic component.
_ap_guess : list of [float, float, float]
    Guess parameters for fitting the aperiodic component.
_ap_bounds : tuple of tuple of float
    Upper and lower bounds on fitting aperiodic component.
_cf_bound : float
    Parameter bounds for center frequency when fitting gaussians.
_bw_std_edge : float
    Bandwidth threshold for edge rejection of peaks, in units of gaussian standard deviation.
_gauss_overlap_thresh : float
    Degree of overlap (in units of standard deviation) between gaussian guesses to drop one.
_gauss_std_limits : list of [float, float]
    Peak width limits, converted to use for gaussian standard deviation parameter.
    This attribute is computed based on `peak_width_limits` and should not be updated directly.
_maxfev : int
    The maximum number of calls to the curve fitting function.
_error_metric : str
    The error metric to use for post-hoc measures of model fit error.
_debug : bool
    Whether the object is set in debug mode.
    This should be controlled by using the `set_debug_mode` method.

Code Notes
----------
Methods without defined docstrings import docs at runtime, from aliased external functions.
"""

import warnings
from copy import deepcopy

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import curve_fit

from fooof.core.items import OBJ_DESC
from fooof.core.info import get_indices
from fooof.core.io import save_fm, load_json
from fooof.core.reports import save_report_fm
from fooof.core.modutils import copy_doc_func_to_method
from fooof.core.utils import group_three, check_array_dim
from fooof.core.funcs import gaussian_function, get_ap_func, infer_ap_func
from fooof.core.errors import (FitError, NoModelError, DataError,
                               NoDataError, InconsistentDataError)
from fooof.core.strings import (gen_settings_str, gen_results_fm_str,
                                gen_issue_str, gen_width_warning_str)

from fooof.plts.fm import plot_fm
from fooof.plts.style import style_spectrum_plot
from fooof.utils.data import trim_spectrum
from fooof.utils.params import compute_gauss_std
from fooof.data import FOOOFResults, FOOOFSettings, FOOOFMetaData
from fooof.sim.gen import gen_freqs, gen_aperiodic, gen_periodic, gen_model


###################################################################################################
###################################################################################################

class FOOOF():
    """Model a physiological power spectrum as a combination of aperiodic and periodic components.

    WARNING: FOOOF expects frequency and power values in linear space.

    Passing in logged frequencies and/or power spectra is not detected,
    and will silently produce incorrect results.

    Parameters
    ----------
    peak_width_limits : tuple of (float, float), optional, default: (0.5, 12.0)
        Limits on possible peak width, in Hz, as (lower_bound, upper_bound).
    max_n_peaks : int, optional, default: inf
        Maximum number of peaks to fit.
    min_peak_height : float, optional, default: 0
        Absolute threshold for detecting peaks, in units of the input data.
    peak_threshold : float, optional, default: 2.0
        Relative threshold for detecting peaks, in units of standard deviation of the input data.
    aperiodic_mode : {'fixed', 'knee'}
        Which approach to take for fitting the aperiodic component.
    verbose : bool, optional, default: True
        Verbosity mode. If True, prints out warnings and general status updates.

    Attributes
    ----------
    freqs : 1d array
        Frequency values for the power spectrum.
    power_spectrum : 1d array
        Power values, stored internally in log10 scale.
    freq_range : list of [float, float]
        Frequency range of the power spectrum, as [lowest_freq, highest_freq].
    freq_res : float
        Frequency resolution of the power spectrum.
    fooofed_spectrum_ : 1d array
        The full model fit of the power spectrum, in log10 scale.
    aperiodic_params_ : 1d array
        Parameters that define the aperiodic fit. As [Offset, (Knee), Exponent].
        The knee parameter is only included if aperiodic component is fit with a knee.
    peak_params_ : 2d array
        Fitted parameter values for the peaks. Each row is a peak, as [CF, PW, BW].
    gaussian_params_ : 2d array
        Parameters that define the gaussian fit(s).
        Each row is a gaussian, as [mean, height, standard deviation].
    r_squared_ : float
        R-squared of the fit between the input power spectrum and the full model fit.
    error_ : float
        Error of the full model fit.
    n_peaks_ : int
        The number of peaks fit in the model.
    has_data : bool
        Whether data is loaded to the object.
    has_model : bool
        Whether model results are available in the object.

    Notes
    -----
    - Commonly used abbreviations used in this module include:
      CF: center frequency, PW: power, BW: Bandwidth, AP: aperiodic
    - Input power spectra must be provided in linear scale.
      Internally they are stored in log10 scale, as this is what the model operates upon.
    - Input power spectra should be smooth, as overly noisy power spectra may lead to bad fits.
      For example, raw FFT inputs are not appropriate. Where possible and appropriate, use
      longer time segments for power spectrum calculation to get smoother power spectra,
      as this will give better model fits.
    - The gaussian params are those that define the gaussian of the fit, where as the peak
      params are a modified version, in which the CF of the peak is the mean of the gaussian,
      the PW of the peak is the height of the gaussian over and above the aperiodic component,
      and the BW of the peak, is 2*std of the gaussian (as 'two sided' bandwidth).
    """
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, peak_width_limits=(0.5, 12.0), max_n_peaks=np.inf, min_peak_height=0.0,
                 peak_threshold=2.0, aperiodic_mode='fixed', verbose=True):
        """Initialize object with desired settings."""

        # Set input settings
        self.peak_width_limits = peak_width_limits
        self.max_n_peaks = max_n_peaks
        self.min_peak_height = min_peak_height
        self.peak_threshold = peak_threshold
        self.aperiodic_mode = aperiodic_mode
        self.verbose = verbose

        ## PRIVATE SETTINGS
        # Percentile threshold, to select points from a flat spectrum for an initial aperiodic fit
        #   Points are selected at a low percentile value to restrict to non-peak points
        self._ap_percentile_thresh = 0.025
        # Guess parameters for aperiodic fitting, [offset, knee, exponent]
        #   If offset guess is None, the first value of the power spectrum is used as offset guess
        #   If exponent guess is None, the abs(log-log slope) of first & last points is used
        self._ap_guess = (None, 0, None)
        # Bounds for aperiodic fitting, as: ((offset_low_bound, knee_low_bound, exp_low_bound),
        #                                    (offset_high_bound, knee_high_bound, exp_high_bound))
        # By default, aperiodic fitting is unbound, but can be restricted here, if desired
        #   Even if fitting without knee, leave bounds for knee (they are dropped later)
        self._ap_bounds = ((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf))
        # Threshold for how far a peak has to be from edge to keep.
        #   This is defined in units of gaussian standard deviation
        self._bw_std_edge = 1.0
        # Degree of overlap between gaussians for one to be dropped
        #   This is defined in units of gaussian standard deviation
        self._gauss_overlap_thresh = 0.75
        # Parameter bounds for center frequency when fitting gaussians, in terms of +/- std dev
        self._cf_bound = 1.5
        # The maximum number of calls to the curve fitting function
        self._maxfev = 5000
        # The error metric to calculate, post model fitting. See `_calc_error` for options
        #   Note: this is used to check error post-hoc, not an objective function for fitting models
        self._error_metric = 'MAE'
        # Set whether in debug mode, in which an error is raised if a model fit fails
        self._debug = False

        # Set internal settings, based on inputs, & initialize data & results attributes
        self._reset_internal_settings()
        self._reset_data_results(True, True, True)


    @property
    def has_data(self):
        """Indicator for if the object contains data."""

        return True if np.any(self.power_spectrum) else False


    @property
    def has_model(self):
        """Indicator for if the object contains a model fit.

        Notes
        -----
        This check uses the aperiodic params, which are:

        - nan if no model has been fit
        - necessarily defined, as floats, if model has been fit
        """

        return True if not np.all(np.isnan(self.aperiodic_params_)) else False


    @property
    def n_peaks_(self):
        """How many peaks were fit in the model."""

        return self.peak_params_.shape[0] if self.has_model else None


    def _reset_internal_settings(self):
        """Set, or reset, internal settings, based on what is provided in init.

        Notes
        -----
        These settings are for internal use, based on what is provided to, or set in `__init__`.
        They should not be altered by the user.
        """

        # Only update these settings if other relevant settings are available
        if self.peak_width_limits:

            # Bandwidth limits are given in 2-sided peak bandwidth
            #   Convert to gaussian std parameter limits
            self._gauss_std_limits = tuple([bwl / 2 for bwl in self.peak_width_limits])
            # Bounds for aperiodic fitting. Drops bounds on knee parameter if not set to fit knee
            self._ap_bounds = self._ap_bounds if self.aperiodic_mode == 'knee' \
                else tuple(bound[0::2] for bound in self._ap_bounds)

        # Otherwise, assume settings are unknown (have been cleared) and set to None
        else:
            self._gauss_std_limits = None
            self._ap_bounds = None


    def _reset_data_results(self, clear_freqs=False, clear_spectrum=False, clear_results=False):
        """Set, or reset, data & results attributes to empty.

        Parameters
        ----------
        clear_freqs : bool, optional, default: False
            Whether to clear frequency attributes.
        clear_spectrum : bool, optional, default: False
            Whether to clear power spectrum attribute.
        clear_results : bool, optional, default: False
            Whether to clear model results attributes.
        """

        if clear_freqs:
            self.freqs = None
            self.freq_range = None
            self.freq_res = None

        if clear_spectrum:
            self.power_spectrum = None

        if clear_results:

            self.aperiodic_params_ = np.array([np.nan] * \
                (2 if self.aperiodic_mode == 'fixed' else 3))
            self.gaussian_params_ = np.empty([0, 3])
            self.peak_params_ = np.empty([0, 3])
            self.r_squared_ = np.nan
            self.error_ = np.nan

            self.fooofed_spectrum_ = None

            self._spectrum_flat = None
            self._spectrum_peak_rm = None
            self._ap_fit = None
            self._peak_fit = None


    def add_data(self, freqs, power_spectrum, freq_range=None):
        """Add data (frequencies, and power spectrum values) to the current object.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power spectrum, in linear space.
        power_spectrum : 1d array
            Power spectrum values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Frequency range to restrict power spectrum to.
            If not provided, keeps the entire range.

        Notes
        -----
        If called on an object with existing data and/or results
        they will be cleared by this method call.
        """

        # If any data is already present, then clear data & results
        #   This is to ensure object consistency of all data & results
        if np.any(self.freqs):
            self._reset_data_results(True, True, True)

        self.freqs, self.power_spectrum, self.freq_range, self.freq_res = \
            self._prepare_data(freqs, power_spectrum, freq_range, 1, self.verbose)


    def add_settings(self, fooof_settings):
        """Add settings into object from a FOOOFSettings object.

        Parameters
        ----------
        fooof_settings : FOOOFSettings
            A data object containing the settings for a FOOOF model.
        """

        for setting in OBJ_DESC['settings']:
            setattr(self, setting, getattr(fooof_settings, setting))

        self._check_loaded_settings(fooof_settings._asdict())


    def add_meta_data(self, fooof_meta_data):
        """Add data information into object from a FOOOFMetaData object.

        Parameters
        ----------
        fooof_meta_data : FOOOFMetaData
            A meta data object containing meta data information.
        """

        for meta_dat in OBJ_DESC['meta_data']:
            setattr(self, meta_dat, getattr(fooof_meta_data, meta_dat))

        self._regenerate_freqs()


    def add_results(self, fooof_result):
        """Add results data into object from a FOOOFResults object.

        Parameters
        ----------
        fooof_result : FOOOFResults
            A data object containing the results from fitting a FOOOF model.
        """

        self.aperiodic_params_ = fooof_result.aperiodic_params
        self.gaussian_params_ = fooof_result.gaussian_params
        self.peak_params_ = fooof_result.peak_params
        self.r_squared_ = fooof_result.r_squared
        self.error_ = fooof_result.error

        self._check_loaded_results(fooof_result._asdict())


    def report(self, freqs=None, power_spectrum=None, freq_range=None, plt_log=False):
        """Run model fit, and display a report, which includes a plot, and printed results.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power spectrum.
        power_spectrum : 1d array, optional
            Power values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Desired frequency range to fit the model to.
            If not provided, fits across the entire given range.
        plt_log : bool, optional, default: False
            Whether or not to plot the frequency axis in log space.

        Notes
        -----
        Data is optional, if data has already been added to the object.
        """

        self.fit(freqs, power_spectrum, freq_range)
        self.plot(plt_log=plt_log)
        self.print_results(concise=False)


    def fit(self, freqs=None, power_spectrum=None, freq_range=None):
        """Fit the full power spectrum as a combination of periodic and aperiodic components.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power spectrum, in linear space.
        power_spectrum : 1d array, optional
            Power values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Frequency range to restrict power spectrum to. If not provided, keeps the entire range.

        Raises
        ------
        NoDataError
            If no data is available to fit.
        FitError
            If model fitting fails to fit. Only raised in debug mode.

        Notes
        -----
        Data is optional, if data has already been added to the object.
        """

        # If freqs & power_spectrum provided together, add data to object.
        if freqs is not None and power_spectrum is not None:
            self.add_data(freqs, power_spectrum, freq_range)
        # If power spectrum provided alone, add to object, and use existing frequency data
        #   Note: be careful passing in power_spectrum data like this:
        #     It assumes the power_spectrum is already logged, with correct freq_range
        elif isinstance(power_spectrum, np.ndarray):
            self.power_spectrum = power_spectrum

        # Check that data is available
        if not self.has_data:
            raise NoDataError("No data available to fit, can not proceed.")

        # Check and warn about width limits (if in verbose mode)
        if self.verbose:
            self._check_width_limits()

        # In rare cases, the model fails to fit, and so uses try / except
        try:

            # Fit the aperiodic component
            self.aperiodic_params_ = self._robust_ap_fit(self.freqs, self.power_spectrum)
            self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_)

            # Flatten the power spectrum using fit aperiodic fit
            self._spectrum_flat = self.power_spectrum - self._ap_fit

            # Find peaks, and fit them with gaussians
            self.gaussian_params_ = self._fit_peaks(np.copy(self._spectrum_flat))

            # Calculate the peak fit
            #   Note: if no peaks are found, this creates a flat (all zero) peak fit
            self._peak_fit = gen_periodic(self.freqs, np.ndarray.flatten(self.gaussian_params_))

            # Create peak-removed (but not flattened) power spectrum
            self._spectrum_peak_rm = self.power_spectrum - self._peak_fit

            # Run final aperiodic fit on peak-removed power spectrum
            #   This overwrites previous aperiodic fit, and recomputes the flattened spectrum
            self.aperiodic_params_ = self._simple_ap_fit(self.freqs, self._spectrum_peak_rm)
            self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_)
            self._spectrum_flat = self.power_spectrum - self._ap_fit

            # Create full power_spectrum model fit
            self.fooofed_spectrum_ = self._peak_fit + self._ap_fit

            # Convert gaussian definitions to peak parameters
            self.peak_params_ = self._create_peak_params(self.gaussian_params_)

            # Calculate R^2 and error of the model fit
            self._calc_r_squared()
            self._calc_error()

        except FitError:

            # If in debug mode, re-raise the error
            if self._debug:
                raise

            # Clear any interim model results that may have run
            #   Partial model results shouldn't be interpreted in light of overall failure
            self._reset_data_results(clear_results=True)

            # Print out status
            if self.verbose:
                print("Model fitting was unsuccessful.")


    def print_settings(self, description=False, concise=False):
        """Print out the current settings.

        Parameters
        ----------
        description : bool, optional, default: False
            Whether to print out a description with current settings.
        concise : bool, optional, default: False
            Whether to print the report in a concise mode, or not.
        """

        print(gen_settings_str(self, description, concise))


    def print_results(self, concise=False):
        """Print out model fitting results.

        Parameters
        ----------
        concise : bool, optional, default: False
            Whether to print the report in a concise mode, or not.
        """

        print(gen_results_fm_str(self, concise))


    @staticmethod
    def print_report_issue(concise=False):
        """Prints instructions on how to report bugs and/or problematic fits.

        Parameters
        ----------
        concise : bool, optional, default: False
            Whether to print the report in a concise mode, or not.
        """

        print(gen_issue_str(concise))


    def get_settings(self):
        """Return user defined settings of the current object.

        Returns
        -------
        FOOOFSettings
            Object containing the settings from the current object.
        """

        return FOOOFSettings(**{key : getattr(self, key) \
                             for key in OBJ_DESC['settings']})


    def get_meta_data(self):
        """Return data information from the current object.

        Returns
        -------
        FOOOFMetaData
            Object containing meta data from the current object.
        """

        return FOOOFMetaData(**{key : getattr(self, key) \
                             for key in OBJ_DESC['meta_data']})


    def get_params(self, name, col=None):
        """Return model fit parameters for specified feature(s).

        Parameters
        ----------
        name : {'aperiodic_params', 'peak_params', 'gaussian_params', 'error', 'r_squared'}
            Name of the data field to extract.
        col : {'CF', 'PW', 'BW', 'offset', 'knee', 'exponent'} or int, optional
            Column name / index to extract from selected data, if requested.
            Only used for name of {'aperiodic_params', 'peak_params', 'gaussian_params'}.

        Returns
        -------
        out : float or 1d array
            Requested data.

        Raises
        ------
        NoModelError
            If there are no model fit parameters available to return.

        Notes
        -----
        For further description of the data you can extract, check the FOOOFResults documentation.

        If there is no data on periodic features, this method will return NaN.
        """

        if not self.has_model:
            raise NoModelError("No model fit results are available to extract, can not proceed.")

        # If col specified as string, get mapping back to integer
        if isinstance(col, str):
            col = get_indices(self.aperiodic_mode)[col]

        # Allow for shortcut alias, without adding `_params`
        if name in ['aperiodic', 'peak', 'gaussian']:
            name = name + '_params'

        # Extract the request data field from object
        out = getattr(self, name + '_')

        # Periodic values can be empty arrays and if so, replace with NaN array
        if isinstance(out, np.ndarray) and out.size == 0:
            out = np.array([np.nan, np.nan, np.nan])

        # Select out a specific column, if requested
        if col is not None:

            # Extract column, & if result is a single value in an array, unpack from array
            out = out[col] if out.ndim == 1 else out[:, col]
            out = out[0] if isinstance(out, np.ndarray) and out.size == 1 else out

        return out


    def get_results(self):
        """Return model fit parameters and goodness of fit metrics.

        Returns
        -------
        FOOOFResults
            Object containing the model fit results from the current object.
        """

        return FOOOFResults(**{key.strip('_') : getattr(self, key) \
            for key in OBJ_DESC['results']})


    @copy_doc_func_to_method(plot_fm)
    def plot(self, plot_peaks=None, plot_aperiodic=True, plt_log=False,
             add_legend=True, save_fig=False, file_name=None, file_path=None,
             ax=None, plot_style=style_spectrum_plot,
             data_kwargs=None, model_kwargs=None, aperiodic_kwargs=None, peak_kwargs=None):

        plot_fm(self, plot_peaks, plot_aperiodic, plt_log, add_legend,
                save_fig, file_name, file_path, ax, plot_style,
                data_kwargs, model_kwargs, aperiodic_kwargs, peak_kwargs)


    # @copy_doc_func_to_method(plot_fm_lin_MG)
    def plot_lin_MG(self, plot_peaks=None, plot_aperiodic=True, plt_log=False,
             add_legend=True, save_fig=False, file_name=None, file_path=None,
             ax=None, plot_style=style_spectrum_plot,
             data_kwargs=None, model_kwargs=None, aperiodic_kwargs=None, peak_kwargs=None,
             label=None):
        
        plot_fm_lin_MG(self, plot_peaks, plot_aperiodic, plt_log, add_legend,
        save_fig, file_name, file_path, ax, plot_style,
        data_kwargs, model_kwargs, aperiodic_kwargs, peak_kwargs, label=label)



    @copy_doc_func_to_method(save_report_fm)
    def save_report(self, file_name, file_path=None, plt_log=False):

        save_report_fm(self, file_name, file_path, plt_log)


    @copy_doc_func_to_method(save_fm)
    def save(self, file_name, file_path=None, append=False,
             save_results=False, save_settings=False, save_data=False):

        save_fm(self, file_name, file_path, append, save_results, save_settings, save_data)


    def load(self, file_name, file_path=None, regenerate=True):
        """Load in a FOOOF formatted JSON file to the current object.

        Parameters
        ----------
        file_name : str or FileObject
            File to load data from.
        file_path : str or None, optional
            Path to directory to load from. If None, loads from current directory.
        regenerate : bool, optional, default: True
            Whether to regenerate the model fit from the loaded data, if data is available.
        """

        # Reset data in object, so old data can't interfere
        self._reset_data_results(True, True, True)

        # Load JSON file, add to self and check loaded data
        data = load_json(file_name, file_path)
        self._add_from_dict(data)
        self._check_loaded_settings(data)
        self._check_loaded_results(data)

        # Regenerate model components, based on what is available
        if regenerate:
            if self.freq_res:
                self._regenerate_freqs()
            if np.all(self.freqs) and np.all(self.aperiodic_params_):
                self._regenerate_model()


    def copy(self):
        """Return a copy of the current object."""

        return deepcopy(self)


    def set_debug_mode(self, debug):
        """Set whether debug mode, wherein an error is raised if fitting is unsuccessful.

        Parameters
        ----------
        debug : bool
            Whether to run in debug mode.
        """

        self._debug = debug


    def _check_width_limits(self):
        """Check and warn about peak width limits / frequency resolution interaction."""

        # Check peak width limits against frequency resolution and warn if too close
        if 1.5 * self.freq_res >= self.peak_width_limits[0]:
            print(gen_width_warning_str(self.freq_res, self.peak_width_limits[0]))


    def _simple_ap_fit(self, freqs, power_spectrum):
        """Fit the aperiodic component of the power spectrum.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        aperiodic_params : 1d array
            Parameter estimates for aperiodic fit.
        """

        # Get the guess parameters and/or calculate from the data, as needed
        #   Note that these are collected as lists, to concatenate with or without knee later
        off_guess = [power_spectrum[0] if not self._ap_guess[0] else self._ap_guess[0]]
        kne_guess = [self._ap_guess[1]] if self.aperiodic_mode == 'knee' else []
        exp_guess = [np.abs(self.power_spectrum[-1] - self.power_spectrum[0] /
                            np.log10(self.freqs[-1]) - np.log10(self.freqs[0]))
                     if not self._ap_guess[2] else self._ap_guess[2]]

        # Collect together guess parameters
        guess = np.array([off_guess + kne_guess + exp_guess])

        # Ignore warnings that are raised in curve_fit
        #   A runtime warning can occur while exploring parameters in curve fitting
        #     This doesn't effect outcome - it won't settle on an answer that does this
        #   It happens if / when b < 0 & |b| > x**2, as it leads to log of a negative number
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                                freqs, power_spectrum, p0=guess,
                                                maxfev=self._maxfev, bounds=self._ap_bounds)
        except RuntimeError:
            raise FitError("Model fitting failed due to not finding parameters in "
                           "the simple aperiodic component fit.")

        return aperiodic_params


    def _robust_ap_fit(self, freqs, power_spectrum):
        """Fit the aperiodic component of the power spectrum robustly, ignoring outliers.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        aperiodic_params : 1d array
            Parameter estimates for aperiodic fit.

        Raises
        ------
        FitError
            If the fitting encounters an error.
        """

        # Do a quick, initial aperiodic fit
        popt = self._simple_ap_fit(freqs, power_spectrum)
        initial_fit = gen_aperiodic(freqs, popt)

        # Flatten power_spectrum based on initial aperiodic fit
        flatspec = power_spectrum - initial_fit

        # Flatten outliers, defined as any points that drop below 0
        flatspec[flatspec < 0] = 0

        # Use percentile threshold, in terms of # of points, to extract and re-fit
        perc_thresh = np.percentile(flatspec, self._ap_percentile_thresh)
        perc_mask = flatspec <= perc_thresh
        freqs_ignore = freqs[perc_mask]
        spectrum_ignore = power_spectrum[perc_mask]

        # Second aperiodic fit - using results of first fit as guess parameters
        #  See note in _simple_ap_fit about warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                                freqs_ignore, spectrum_ignore, p0=popt,
                                                maxfev=self._maxfev, bounds=self._ap_bounds)
        except RuntimeError:
            raise FitError("Model fitting failed due to not finding "
                           "parameters in the robust aperiodic fit.")
        except TypeError:
            raise FitError("Model fitting failed due to sub-sampling in the robust aperiodic fit.")

        return aperiodic_params


    def _fit_peaks(self, flat_iter):
        """Iteratively fit peaks to flattened spectrum.

        Parameters
        ----------
        flat_iter : 1d array
            Flattened power spectrum values.

        Returns
        -------
        gaussian_params : 2d array
            Parameters that define the gaussian fit(s).
            Each row is a gaussian, as [mean, height, standard deviation].
        """

        # Initialize matrix of guess parameters for gaussian fitting
        guess = np.empty([0, 3])

        # Find peak: Loop through, finding a candidate peak, and fitting with a guess gaussian
        #   Stopping procedures: limit on # of peaks, or relative or absolute height thresholds
        while len(guess) < self.max_n_peaks:

            # Find candidate peak - the maximum point of the flattened spectrum
            max_ind = np.argmax(flat_iter)
            max_height = flat_iter[max_ind]

            # Stop searching for peaks once height drops below height threshold
            if max_height <= self.peak_threshold * np.std(flat_iter):
                break

            # Set the guess parameters for gaussian fitting, specifying the mean and height
            guess_freq = self.freqs[max_ind]
            guess_height = max_height

            # Halt fitting process if candidate peak drops below minimum height
            if not guess_height > self.min_peak_height:
                break

            # Data-driven first guess at standard deviation
            #   Find half height index on each side of the center frequency
            half_height = 0.5 * max_height
            le_ind = next((val for val in range(max_ind - 1, 0, -1)
                           if flat_iter[val] <= half_height), None)
            ri_ind = next((val for val in range(max_ind + 1, len(flat_iter), 1)
                           if flat_iter[val] <= half_height), None)

            # Guess bandwidth procedure: estimate the width of the peak
            try:
                # Get an estimated width from the shortest side of the peak
                #   We grab shortest to avoid estimating very large values from overlapping peaks
                # Grab the shortest side, ignoring a side if the half max was not found
                short_side = min([abs(ind - max_ind) \
                    for ind in [le_ind, ri_ind] if ind is not None])

                # Use the shortest side to estimate full-width, half max (converted to Hz)
                #   and use this to estimate that guess for gaussian standard deviation
                fwhm = short_side * 2 * self.freq_res
                guess_std = compute_gauss_std(fwhm)

            except ValueError:
                # This procedure can fail (extremely rarely), if both le & ri ind's end up as None
                #   In this case, default the guess to the average of the peak width limits
                guess_std = np.mean(self.peak_width_limits)

            # Check that guess value isn't outside preset limits - restrict if so
            #   Note: without this, curve_fitting fails if given guess > or < bounds
            if guess_std < self._gauss_std_limits[0]:
                guess_std = self._gauss_std_limits[0]
            if guess_std > self._gauss_std_limits[1]:
                guess_std = self._gauss_std_limits[1]

            # Collect guess parameters and subtract this guess gaussian from the data
            guess = np.vstack((guess, (guess_freq, guess_height, guess_std)))
            peak_gauss = gaussian_function(self.freqs, guess_freq, guess_height, guess_std)
            flat_iter = flat_iter - peak_gauss

        # Check peaks based on edges, and on overlap, dropping any that violate requirements
        guess = self._drop_peak_cf(guess)
        guess = self._drop_peak_overlap(guess)

        # If there are peak guesses, fit the peaks, and sort results
        if len(guess) > 0:
            gaussian_params = self._fit_peak_guess(guess)
            gaussian_params = gaussian_params[gaussian_params[:, 0].argsort()]
        else:
            gaussian_params = np.empty([0, 3])

        return gaussian_params


    def _fit_peak_guess(self, guess):
        """Fits a group of peak guesses with a fit function.

        Parameters
        ----------
        guess : 2d array, shape=[n_peaks, 3]
            Guess parameters for gaussian fits to peaks, as gaussian parameters.

        Returns
        -------
        gaussian_params : 2d array, shape=[n_peaks, 3]
            Parameters for gaussian fits to peaks, as gaussian parameters.
        """

        # Set the bounds for CF, enforce positive height value, and set bandwidth limits
        #   Note that 'guess' is in terms of gaussian std, so +/- BW is 2 * the guess_gauss_std
        #   This set of list comprehensions is a way to end up with bounds in the form:
        #     ((cf_low_peak1, height_low_peak1, bw_low_peak1, *repeated for n_peaks*),
        #      (cf_high_peak1, height_high_peak1, bw_high_peak, *repeated for n_peaks*))
        #     ^where each value sets the bound on the specified parameter
        lo_bound = [[peak[0] - 2 * self._cf_bound * peak[2], 0, self._gauss_std_limits[0]]
                    for peak in guess]
        hi_bound = [[peak[0] + 2 * self._cf_bound * peak[2], np.inf, self._gauss_std_limits[1]]
                    for peak in guess]

        # Check that CF bounds are within frequency range
        #   If they are  not, update them to be restricted to frequency range
        lo_bound = [bound if bound[0] > self.freq_range[0] else \
            [self.freq_range[0], *bound[1:]] for bound in lo_bound]
        hi_bound = [bound if bound[0] < self.freq_range[1] else \
            [self.freq_range[1], *bound[1:]] for bound in hi_bound]

        # Unpacks the embedded lists into flat tuples
        #   This is what the fit function requires as input
        gaus_param_bounds = (tuple([item for sublist in lo_bound for item in sublist]),
                             tuple([item for sublist in hi_bound for item in sublist]))

        # Flatten guess, for use with curve fit
        guess = np.ndarray.flatten(guess)

        # Fit the peaks
        try:
            gaussian_params, _ = curve_fit(gaussian_function, self.freqs, self._spectrum_flat,
                                           p0=guess, maxfev=self._maxfev, bounds=gaus_param_bounds)
        except RuntimeError:
            raise FitError("Model fitting failed due to not finding "
                           "parameters in the peak component fit.")
        except LinAlgError:
            raise FitError("Model fitting failed due to a LinAlgError during peak fitting. "
                           "This can happen with settings that are too liberal, leading, "
                           "to a large number of guess peaks that cannot be fit together.")

        # Re-organize params into 2d matrix
        gaussian_params = np.array(group_three(gaussian_params))

        return gaussian_params


    def _create_peak_params(self, gaus_params):
        """Copies over the gaussian params to peak outputs, updating as appropriate.

        Parameters
        ----------
        gaus_params : 2d array
            Parameters that define the gaussian fit(s), as gaussian parameters.

        Returns
        -------
        peak_params : 2d array
            Fitted parameter values for the peaks, with each row as [CF, PW, BW].

        Notes
        -----
        The gaussian center is unchanged as the peak center frequency.

        The gaussian height is updated to reflect the height of the peak above
        the aperiodic fit. This is returned instead of the gaussian height, as
        the gaussian height is harder to interpret, due to peak overlaps.

        The gaussian standard deviation is updated to be 'both-sided', to reflect the
        'bandwidth' of the peak, as opposed to the gaussian parameter, which is 1-sided.

        Performing this conversion requires that the model has been run,
        with `freqs`, `fooofed_spectrum_` and `_ap_fit` all required to be available.
        """

        peak_params = np.empty([0, 3])

        for ii, peak in enumerate(gaus_params):

            # Gets the index of the power_spectrum at the frequency closest to the CF of the peak
            ind = min(range(len(self.freqs)), key=lambda ii: abs(self.freqs[ii] - peak[0]))

            # Collect peak parameter data
            peak_params = np.vstack((peak_params,
                                     [peak[0],
                                      self.fooofed_spectrum_[ind] - self._ap_fit[ind],
                                      peak[2] * 2]))

        return peak_params


    def _drop_peak_cf(self, guess):
        """Check whether to drop peaks based on center's proximity to the edge of the spectrum.

        Parameters
        ----------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].

        Returns
        -------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].
        """

        cf_params = [item[0] for item in guess]
        bw_params = [item[2] * self._bw_std_edge for item in guess]

        # Check if peaks within drop threshold from the edge of the frequency range
        keep_peak = \
            (np.abs(np.subtract(cf_params, self.freq_range[0])) > bw_params) & \
            (np.abs(np.subtract(cf_params, self.freq_range[1])) > bw_params)

        # Drop peaks that fail the center frequency edge criterion
        guess = np.array([gu for (gu, keep) in zip(guess, keep_peak) if keep])

        return guess


    def _drop_peak_overlap(self, guess):
        """Checks whether to drop gaussians based on amount of overlap.

        Parameters
        ----------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].

        Returns
        -------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].

        Notes
        -----
        For any gaussians with an overlap that crosses the threshold,
        the lowest height guess guassian is dropped.
        """

        # Sort the peak guesses by increasing frequency, so adjacenent peaks can
        #   be compared from right to left.
        guess = sorted(guess, key=lambda x: float(x[0]))

        # Calculate standard deviation bounds for checking amount of overlap
        #   The bounds are the gaussian frequncy +/- gaussian standard deviation
        bounds = [[peak[0] - peak[2] * self._gauss_overlap_thresh,
                   peak[0] + peak[2] * self._gauss_overlap_thresh] for peak in guess]

        # Loop through peak bounds, comparing current bound to that of next peak
        #   If the left peak's upper bound extends pass the right peaks lower bound,
        #   Then drop the guassian with the lower height.
        drop_inds = []
        for ind, b_0 in enumerate(bounds[:-1]):
            b_1 = bounds[ind + 1]

            # Check if bound of current peak extends into next peak
            if b_0[1] > b_1[0]:

                # If so, get the index of the gaussian with the lowest height (to drop)
                drop_inds.append([ind, ind + 1][np.argmin([guess[ind][1], guess[ind + 1][1]])])

        # Drop any peaks guesses that overlap too much, based on threshold
        keep_peak = [not ind in drop_inds for ind in range(len(guess))]
        guess = np.array([gu for (gu, keep) in zip(guess, keep_peak) if keep])

        return guess


    def _calc_r_squared(self):
        """Calculate the r-squared goodness of fit of the model, compared to the original data."""

        r_val = np.corrcoef(self.power_spectrum, self.fooofed_spectrum_)
        self.r_squared_ = r_val[0][1] ** 2


    def _calc_error(self, metric=None):
        """Calculate the overall error of the model fit, compared to the original data.

        Parameters
        ----------
        metric : {'MAE', 'MSE', 'RMSE'}, optional
            Which error measure to calculate.

        Raises
        ------
        ValueError
            If the requested error metric is not understood.

        Notes
        -----
        Which measure is applied is by default controlled by the `_error_metric` attribute.
        """

        # If metric is not specified, use the default approach
        metric = self._error_metric if not metric else metric

        if metric == 'MAE':
            self.error_ = np.abs(self.power_spectrum - self.fooofed_spectrum_).mean()

        elif metric == 'MSE':
            self.error_ = ((self.power_spectrum - self.fooofed_spectrum_) ** 2).mean()

        elif metric == 'RMSE':
            self.error_ = np.sqrt(((self.power_spectrum - self.fooofed_spectrum_) ** 2).mean())

        else:
            msg = "Error metric '{}' not understood or not implemented.".format(metric)
            raise ValueError(msg)


    @staticmethod
    def _prepare_data(freqs, power_spectrum, freq_range, spectra_dim=1, verbose=True):
        """Prepare input data for adding to current object.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear space.
        power_spectrum : 1d or 2d array
            Power values, which must be input in linear space.
            1d vector, or 2d as [n_power_spectra, n_freqs].
        freq_range : list of [float, float]
            Frequency range to restrict power spectrum to. If None, keeps the entire range.
        spectra_dim : int, optional, default: 1
            Dimensionality that the power spectra should have.
        verbose : bool, optional
            Whether to be verbose in printing out warnings.

        Returns
        -------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear space.
        power_spectrum : 1d or 2d array
            Power spectrum values, in log10 scale.
            1d vector, or 2d as [n_power_specta, n_freqs].
        freq_range : list of [float, float]
            Minimum and maximum values of the frequency vector.
        freq_res : float
            Frequency resolution of the power spectrum.

        Raises
        ------
        DataError
            If there is an issue with the data.
        InconsistentDataError
            If the input data are inconsistent size.
        """

        # Check that data are the right types
        if not isinstance(freqs, np.ndarray) or not isinstance(power_spectrum, np.ndarray):
            raise DataError("Input data must be numpy arrays.")

        # Check that data have the right dimensionality
        if freqs.ndim != 1 or (power_spectrum.ndim != spectra_dim):
            raise DataError("Inputs are not the right dimensions.")

        # Check that data sizes are compatible
        if freqs.shape[-1] != power_spectrum.shape[-1]:
            raise InconsistentDataError("The input frequencies and power spectra "
                                        "are not consistent size.")

        # Check if power values are complex
        if np.iscomplexobj(power_spectrum):
            raise DataError("Input power spectra are complex values. "
                            "FOOOF does not currently support complex inputs.")

        # Force data to be dtype of float64
        #   If they end up as float32, or less, scipy curve_fit fails (sometimes implicitly)
        if freqs.dtype != 'float64':
            freqs = freqs.astype('float64')
        if power_spectrum.dtype != 'float64':
            power_spectrum = power_spectrum.astype('float64')

        # Check frequency range, trim the power_spectrum range if requested
        if freq_range:
            freqs, power_spectrum = trim_spectrum(freqs, power_spectrum, freq_range)

        # Check if freqs start at 0 and move up one value if so
        #   Aperiodic fit gets an inf if freq of 0 is included, which leads to an error
        if freqs[0] == 0.0:
            freqs, power_spectrum = trim_spectrum(freqs, power_spectrum, [freqs[1], freqs.max()])
            if verbose:
                print("\nFOOOF WARNING: Skipping frequency == 0, "
                      "as this causes a problem with fitting.")

        # Calculate frequency resolution, and actual frequency range of the data
        freq_range = [freqs.min(), freqs.max()]
        freq_res = freqs[1] - freqs[0]

        # Log power values
        power_spectrum = np.log10(power_spectrum)

        # Check if there are any infs / nans, and raise an error if so
        if np.any(np.isinf(power_spectrum)) or np.any(np.isnan(power_spectrum)):
            raise DataError("The input power spectra data, after logging, contains NaNs or Infs. "
                            "This will cause the fitting to fail. "
                            "One reason this can happen is if inputs are already logged. "
                            "Inputs data should be in linear spacing, not log.")

        return freqs, power_spectrum, freq_range, freq_res


    def _add_from_dict(self, data):
        """Add data to object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary of data to add to self.
        """

        # Reconstruct object from loaded data
        for key in data.keys():
            setattr(self, key, data[key])


    def _check_loaded_results(self, data):
        """Check if results have been added and check data.

        Parameters
        ----------
        data : dict
            A dictionary of data that has been added to the object.
        """

        # If results loaded, check dimensions of peak parameters
        #   This fixes an issue where they end up the wrong shape if they are empty (no peaks)
        if set(OBJ_DESC['results']).issubset(set(data.keys())):
            self.peak_params_ = check_array_dim(self.peak_params_)
            self.gaussian_params_ = check_array_dim(self.gaussian_params_)


    def _check_loaded_settings(self, data):
        """Check if settings added, and update the object as needed.

        Parameters
        ----------
        data : dict
            A dictionary of data that has been added to the object.
        """

        # If settings not loaded from file, clear from object, so that default
        # settings, which are potentially wrong for loaded data, aren't kept
        if not set(OBJ_DESC['settings']).issubset(set(data.keys())):

            # Reset all public settings to None
            for setting in OBJ_DESC['settings']:
                setattr(self, setting, None)

            # If aperiodic params available, infer whether knee fitting was used,
            if not np.all(np.isnan(self.aperiodic_params_)):
                self.aperiodic_mode = infer_ap_func(self.aperiodic_params_)

        # Reset internal settings so that they are consistent with what was loaded
        #   Note that this will set internal settings to None, if public settings unavailable
        self._reset_internal_settings()


    def _regenerate_freqs(self):
        """Regenerate the frequency vector, given the object metadata."""

        self.freqs = gen_freqs(self.freq_range, self.freq_res)


    def _regenerate_model(self):
        """Regenerate model fit from parameters."""

        self.fooofed_spectrum_, self._peak_fit, self._ap_fit = gen_model(
            self.freqs, self.aperiodic_params_, self.gaussian_params_, return_components=True)


# MODIFIED MG
from fooof.plts.spectra import plot_spectrum
from fooof.plts.settings import PLT_FIGSIZES, PLT_COLORS
from fooof.plts.utils import check_ax, check_plot_kwargs
from fooof.plts.style import check_n_style, style_spectrum_plot



def plot_fm_lin_MG(fm, plot_peaks=None, plot_aperiodic=True, plt_log=False, add_legend=True,
            save_fig=False, file_name=None, file_path=None,
            ax=None, plot_style=style_spectrum_plot,
            data_kwargs=None, model_kwargs=None, aperiodic_kwargs=None, peak_kwargs=None,
            label='Fit +\nOscillatory PSD'):
            """Plot the power spectrum and model fit results from a FOOOF object.
        
            Parameters
            ----------
            fm : FOOOF
                Object containing a power spectrum and (optionally) results from fitting.
            plot_peaks : None or {'shade', 'dot', 'outline', 'line'}, optional
                What kind of approach to take to plot peaks. If None, peaks are not specifically plotted.
                Can also be a combination of approaches, separated by '-', for example: 'shade-line'.
            plot_aperiodic : boolean, optional, default: True
                Whether to plot the aperiodic component of the model fit.
            plt_log : boolean, optional, default: False
                Whether to plot the frequency values in log10 spacing.
            add_legend : boolean, optional, default: False
                Whether to add a legend describing the plot components.
            save_fig : bool, optional, default: False
                Whether to save out a copy of the plot.
            file_name : str, optional
                Name to give the saved out file.
            file_path : str, optional
                Path to directory to save to. If None, saves to current directory.
            ax : matplotlib.Axes, optional
                Figure axes upon which to plot.
            plot_style : callable, optional, default: style_spectrum_plot
                A function to call to apply styling & aesthetics to the plot.
            data_kwargs, model_kwargs, aperiodic_kwargs, peak_kwargs : None or dict, optional
                Keyword arguments to pass into the plot call for each plot element.
        
            Notes
            -----
            Since FOOOF objects store power values in log spacing,
            the y-axis (power) is plotted in log spacing by default.
            """
        
            ax = check_ax(ax, PLT_FIGSIZES['spectral'])
        
            # Log settings - note that power values in FOOOF objects are already logged
            log_freqs = plt_log
            log_powers = False
        
            # Plot the data, if available
            if fm.has_data:
                data_kwargs = check_plot_kwargs(data_kwargs, \
                    {'color' : PLT_COLORS['data'], 'linewidth' : 2.0,
                     'label' : None})
                plot_spectrum(fm.freqs, 10**fm.power_spectrum, log_freqs, log_powers,
                              ax=ax, plot_style=None, **data_kwargs)
        
            # Add the full model fit, and components (if requested)
            if fm.has_model:
                model_kwargs = check_plot_kwargs(model_kwargs, \
                    {'color' : PLT_COLORS['model'], 'linewidth' : 3.0, 'alpha' : 0.5,
                     'label' : label if add_legend else None})
                plot_spectrum(fm.freqs, 10**fm.fooofed_spectrum_, log_freqs, log_powers,
                              ax=ax, plot_style=None, **model_kwargs)
        
                # Plot the aperiodic component of the model fit
                if plot_aperiodic:
                    aperiodic_kwargs = check_plot_kwargs(aperiodic_kwargs, \
                        {'color' : PLT_COLORS['aperiodic'], 'linewidth' : 3.0, 'alpha' : 0.5,
                         'linestyle' : 'dashed', 'label' : None})
                    plot_spectrum(fm.freqs, 10**fm._ap_fit, log_freqs, log_powers,
                                  ax=ax, plot_style=None, **aperiodic_kwargs)
        
            # Apply style to plot
            check_n_style(plot_style, ax, log_freqs, True)


"""Fooof script modified by Moritz Gerster.

Plots for annotating power spectrum fittings and models."""

import numpy as np

from fooof.core.utils import nearest_ind
from fooof.core.errors import NoModelError
from fooof.core.funcs import gaussian_function
from fooof.core.modutils import safe_import, check_dependency
from fooof.sim.gen import gen_aperiodic
from fooof.plts.utils import check_ax
from fooof.plts.spectra import plot_spectrum
from fooof.plts.settings import PLT_FIGSIZES, PLT_COLORS
from fooof.plts.style import check_n_style, style_spectrum_plot
from fooof.analysis.periodic import get_band_peak_fm
from fooof.utils.params import compute_knee_frequency, compute_fwhm

plt = safe_import('.pyplot', 'matplotlib')
mpatches = safe_import('.patches', 'matplotlib')

###################################################################################################
###################################################################################################

@check_dependency(plt, 'matplotlib')
def plot_annotated_peak_search(fm, plot_style=style_spectrum_plot):
    """Plot a series of plots illustrating the peak search from a flattened spectrum.

    Parameters
    ----------
    fm : FOOOF
        FOOOF object, with model fit, data and settings available.
    plot_style : callable, optional, default: style_spectrum_plot
        A function to call to apply styling & aesthetics to the plots.
    """

    # Recalculate the initial aperiodic fit and flattened spectrum that
    #   is the same as the one that is used in the peak fitting procedure
    flatspec = fm.power_spectrum - \
        gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))

    # Calculate ylims of the plot that are scaled to the range of the data
    ylims = [min(flatspec) - 0.1 * np.abs(min(flatspec)), max(flatspec) + 0.1 * max(flatspec)]

    # Loop through the iterative search for each peak
    for ind in range(fm.n_peaks_ + 1):

        # This forces the creation of a new plotting axes per iteration
        ax = check_ax(None, PLT_FIGSIZES['spectral'])

        plot_spectrum(fm.freqs, flatspec, ax=ax, plot_style=None,
                      label='Flattened Spectrum', color=PLT_COLORS['data'], linewidth=2.5)
        plot_spectrum(fm.freqs, [fm.peak_threshold * np.std(flatspec)]*len(fm.freqs),
                      ax=ax, plot_style=None, label='Relative Threshold',
                      color='orange', linewidth=2.5, linestyle='dashed')
        plot_spectrum(fm.freqs, [fm.min_peak_height]*len(fm.freqs),
                      ax=ax, plot_style=None, label='Absolute Threshold',
                      color='red', linewidth=2.5, linestyle='dashed')

        maxi = np.argmax(flatspec)
        ax.plot(fm.freqs[maxi], flatspec[maxi], '.',
                color=PLT_COLORS['periodic'], alpha=0.75, markersize=30)

        ax.set_ylim(ylims)
        ax.set_title('Iteration #' + str(ind+1), fontsize=16)

        if ind < fm.n_peaks_:

            gauss = gaussian_function(fm.freqs, *fm.gaussian_params_[ind, :])
            plot_spectrum(fm.freqs, gauss, ax=ax, plot_style=None,
                          label='Gaussian Fit', color=PLT_COLORS['periodic'],
                          linestyle=':', linewidth=3.0)

            flatspec = flatspec - gauss

        check_n_style(plot_style, ax, False, True)


@check_dependency(plt, 'matplotlib')
def plot_annotated_peak_search_MG(fm, ind_max, ax, c_flat="k", c_thresh="orange", c_gauss="g",
                                  plot_style=style_spectrum_plot,
                                  label_flat="Flattened PSD",
                                  label_rthresh="standard deviation",
                                  label_gauss="Gaussian fit",
                                  lw=1, markersize=10,
                                  anno_rthresh_font=None):
    """Plot a series of plots illustrating the peak search from a flattened spectrum.

    Parameters
    ----------
    fm : FOOOF
        FOOOF object, with model fit, data and settings available.
    plot_style : callable, optional, default: style_spectrum_plot
        A function to call to apply styling & aesthetics to the plots.
    """

    # Recalculate the initial aperiodic fit and flattened spectrum that
    #   is the same as the one that is used in the peak fitting procedure
    flatspec = fm.power_spectrum - \
        gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    #flatspec = 10**fm.power_spectrum - \
     #   10**gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))

    # Calculate ylims of the plot that are scaled to the range of the data
    # ylims = [min(10**flatspec) - 0.1 * np.abs(min(10**flatspec)), max(10**flatspec) + 0.1 * max(10**flatspec)]

    # Loop through the iterative search for each peak
    for ind in range(fm.n_peaks_ + 1):

        # This forces the creation of a new plotting axes per iteration
        # ax = check_ax(None, PLT_FIGSIZES['spectral'])
        
        if ind == ind_max:
        
            plot_spectrum(fm.freqs, flatspec, ax=ax, plot_style=None,
                          color=c_flat, linewidth=lw, label=label_flat)
            rthresh = fm.peak_threshold * np.std(flatspec)
            plot_spectrum(fm.freqs, np.array([rthresh]*len(fm.freqs)),
                          ax=ax, plot_style=None, label=label_rthresh,
                          color=c_thresh, linewidth=lw/1.5,
                          linestyle='dashed')
            if anno_rthresh_font:
                ax.annotate(f"{fm.peak_threshold:.0f}"r"$\cdot$SD",
                            xy=(fm.freqs[-1], rthresh), ha="left",
                            va="center",
                            xytext=(fm.freqs[-1], rthresh),
                            annotation_clip=False,
                            fontsize=anno_rthresh_font)
# =============================================================================
#             plot_spectrum(fm.freqs, [fm.min_peak_height]*len(fm.freqs),
#                           ax=ax, plot_style=None, label='Absolute Threshold',
#                           color='red', linewidth=lw, linestyle='dashed')
# =============================================================================
            #ax.set_yticks([])
            #ax.set_yticklabels([])
    
        # maxi = np.argmax(flatspec)
        
# =============================================================================
#         if ind == ind_max:
#             ax.plot(fm.freqs[maxi], flatspec[maxi], '.',
#                     color=c_gauss, alpha=0.75, markersize=markersize)
# =============================================================================
            #ax.set_yticks([])
            #ax.set_yticklabels([])
        
            # ax.set_ylim(ylims)
            # ax.set_title('Iteration #' + str(ind+1), fontsize=16)
    
        if ind < fm.n_peaks_:
    
            gauss = gaussian_function(fm.freqs, *fm.gaussian_params_[ind, :])
            
            if ind == ind_max:
                    
                plot_spectrum(fm.freqs, gauss, ax=ax, plot_style=None,
                              label=label_gauss, color=c_gauss,
                              linestyle=':', linewidth=1.5*lw)
                #ax.set_yticks([])
                #ax.set_yticklabels([])
    
            flatspec = flatspec - gauss


@check_dependency(plt, 'matplotlib')
def plot_annotated_model(fm, plt_log=False, annotate_peaks=True, annotate_aperiodic=True,
                         ax=None, plot_style=style_spectrum_plot):
    """Plot a an annotated power spectrum and model, from a FOOOF object.

    Parameters
    ----------
    fm : FOOOF
        FOOOF object, with model fit, data and settings available.
    plt_log : boolean, optional, default: False
        Whether to plot the frequency values in log10 spacing.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    plot_style : callable, optional, default: style_spectrum_plot
        A function to call to apply styling & aesthetics to the plots.

    Raises
    ------
    NoModelError
        If there are no model results available to plot.
    """

    # Check that model is available
    if not fm.has_model:
        raise NoModelError("No model is available to plot, can not proceed.")

    # Settings
    fontsize = 15
    lw1 = 4.0
    lw2 = 3.0
    ms1 = 12

    # Create the baseline figure
    ax = check_ax(ax, PLT_FIGSIZES['spectral'])
    fm.plot(plot_peaks='dot-shade-width', plt_log=plt_log, ax=ax, plot_style=None,
            data_kwargs={'lw' : lw1, 'alpha' : 0.6},
            aperiodic_kwargs={'lw' : lw1, 'zorder' : 10},
            model_kwargs={'lw' : lw1, 'alpha' : 0.5},
            peak_kwargs={'dot' : {'color' : PLT_COLORS['periodic'], 'ms' : ms1, 'lw' : lw2},
                         'shade' : {'color' : PLT_COLORS['periodic']},
                         'width' : {'color' : PLT_COLORS['periodic'], 'alpha' : 0.75, 'lw' : lw2}})

    # Get freqs for plotting, and convert to log if needed
    freqs = fm.freqs if not plt_log else np.log10(fm.freqs)

    ## Buffers: for spacing things out on the plot (scaled by plot values)
    x_buff1 = max(freqs) * 0.1
    x_buff2 = max(freqs) * 0.25
    y_buff1 = 0.15 * np.ptp(ax.get_ylim())
    shrink = 0.1

    # There is a bug in annotations for some perpendicular lines, so add small offset
    #   See: https://github.com/matplotlib/matplotlib/issues/12820. Fixed in 3.2.1.
    bug_buff = 0.000001

    if annotate_peaks:

        # Extract largest peak, to annotate, grabbing gaussian params
        gauss = get_band_peak_fm(fm, fm.freq_range, attribute='gaussian_params')

        peak_ctr, peak_hgt, peak_wid = gauss
        bw_freqs = [peak_ctr - 0.5 * compute_fwhm(peak_wid),
                    peak_ctr + 0.5 * compute_fwhm(peak_wid)]

        if plt_log:
            peak_ctr = np.log10(peak_ctr)
            bw_freqs = np.log10(bw_freqs)

        peak_top = fm.power_spectrum[nearest_ind(freqs, peak_ctr)]

        # Annotate Peak CF
        ax.annotate('Center Frequency',
                    xy=(peak_ctr, peak_top),
                    xytext=(peak_ctr, peak_top+np.abs(0.6*peak_hgt)),
                    verticalalignment='center',
                    horizontalalignment='center',
                    arrowprops=dict(facecolor=PLT_COLORS['periodic'], shrink=shrink),
                    color=PLT_COLORS['periodic'], fontsize=fontsize)

        # Annotate Peak PW
        ax.annotate('Power',
                    xy=(peak_ctr, peak_top-0.3*peak_hgt),
                    xytext=(peak_ctr+x_buff1, peak_top-0.3*peak_hgt),
                    verticalalignment='center',
                    arrowprops=dict(facecolor=PLT_COLORS['periodic'], shrink=shrink),
                    color=PLT_COLORS['periodic'], fontsize=fontsize)

        # Annotate Peak BW
        bw_buff = (peak_ctr - bw_freqs[0])/2
        ax.annotate('Bandwidth',
                    xy=(peak_ctr-bw_buff+bug_buff, peak_top-(0.5*peak_hgt)),
                    xytext=(peak_ctr-bw_buff, peak_top-(1.5*peak_hgt)),
                    verticalalignment='center',
                    horizontalalignment='right',
                    arrowprops=dict(facecolor=PLT_COLORS['periodic'], shrink=shrink),
                    color=PLT_COLORS['periodic'], fontsize=fontsize, zorder=20)

    if annotate_aperiodic:

        # Annotate Aperiodic Offset
        #   Add a line to indicate offset, without adjusting plot limits below it
        ax.set_autoscaley_on(False)
        ax.plot([freqs[0], freqs[0]], [ax.get_ylim()[0], fm.fooofed_spectrum_[0]],
                color=PLT_COLORS['aperiodic'], linewidth=lw2, alpha=0.5)
        ax.annotate('Offset',
                    xy=(freqs[0]+bug_buff, fm.power_spectrum[0]-y_buff1),
                    xytext=(freqs[0]-x_buff1, fm.power_spectrum[0]-y_buff1),
                    verticalalignment='center',
                    horizontalalignment='center',
                    arrowprops=dict(facecolor=PLT_COLORS['aperiodic'], shrink=shrink),
                    color=PLT_COLORS['aperiodic'], fontsize=fontsize)

        # Annotate Aperiodic Knee
        if fm.aperiodic_mode == 'knee':

            # Find the knee frequency point to annotate
            knee_freq = compute_knee_frequency(fm.get_params('aperiodic', 'knee'),
                                               fm.get_params('aperiodic', 'exponent'))
            knee_freq = np.log10(knee_freq) if plt_log else knee_freq
            knee_pow = fm.power_spectrum[nearest_ind(freqs, knee_freq)]

            # Add a dot to the plot indicating the knee frequency
            ax.plot(knee_freq, knee_pow, 'o', color=PLT_COLORS['aperiodic'], ms=ms1*1.5, alpha=0.7)

            ax.annotate('Knee',
                        xy=(knee_freq, knee_pow),
                        xytext=(knee_freq-x_buff2, knee_pow-y_buff1),
                        verticalalignment='center',
                        arrowprops=dict(facecolor=PLT_COLORS['aperiodic'], shrink=shrink),
                        color=PLT_COLORS['aperiodic'], fontsize=fontsize)

        # Annotate Aperiodic Exponent
        mid_ind = int(len(freqs)/2)
        ax.annotate('Exponent',
                    xy=(freqs[mid_ind], fm.power_spectrum[mid_ind]),
                    xytext=(freqs[mid_ind]-x_buff2, fm.power_spectrum[mid_ind]-y_buff1),
                    verticalalignment='center',
                    arrowprops=dict(facecolor=PLT_COLORS['aperiodic'], shrink=shrink),
                    color=PLT_COLORS['aperiodic'], fontsize=fontsize)

    # Apply style to plot & tune grid styling
    check_n_style(plot_style, ax, plt_log, True)
    ax.grid(True, alpha=0.5)

    # Add labels to plot in the legend
    da_patch = mpatches.Patch(color=PLT_COLORS['data'], label='Original Data')
    ap_patch = mpatches.Patch(color=PLT_COLORS['aperiodic'], label='Aperiodic Parameters')
    pe_patch = mpatches.Patch(color=PLT_COLORS['periodic'], label='Peak Parameters')
    mo_patch = mpatches.Patch(color=PLT_COLORS['model'], label='Full Model')

    handles = [da_patch, ap_patch if annotate_aperiodic else None,
               pe_patch if annotate_peaks else None, mo_patch]
    handles = [el for el in handles if el is not None]

    ax.legend(handles=handles, handlelength=1, fontsize='x-large')
