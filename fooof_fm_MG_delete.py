#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:01:36 2021

@author: moritzgerster
"""
"""Plots for the FOOOF object.

Notes
-----
This file contains plotting functions that take as input a FOOOF object.
"""

import numpy as np

from fooof.core.io import fname, fpath
from fooof.core.utils import nearest_ind
from fooof.core.modutils import safe_import, check_dependency
from fooof.sim.gen import gen_periodic
from fooof.utils.data import trim_spectrum
from fooof.utils.params import compute_fwhm
from fooof.plts.spectra import plot_spectrum
from fooof.plts.settings import PLT_FIGSIZES, PLT_COLORS
from fooof.plts.utils import check_ax, check_plot_kwargs
from fooof.plts.style import check_n_style, style_spectrum_plot

plt = safe_import('.pyplot', 'matplotlib')

###################################################################################################
###################################################################################################



@check_dependency(plt, 'matplotlib')
def plot_fm_lin_MG(fm, plot_peaks=None, plot_aperiodic=True, plt_log=False, add_legend=True,
            save_fig=False, file_name=None, file_path=None,
            ax=None, plot_style=style_spectrum_plot,
            data_kwargs=None, model_kwargs=None, aperiodic_kwargs=None, peak_kwargs=None):
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
             'label' : 'Original Spectrum' if add_legend else None})
        plot_spectrum(fm.freqs, 10**fm.power_spectrum, log_freqs, log_powers,
                      ax=ax, plot_style=None, **data_kwargs)

    # Add the full model fit, and components (if requested)
    if fm.has_model:
        model_kwargs = check_plot_kwargs(model_kwargs, \
            {'color' : PLT_COLORS['model'], 'linewidth' : 3.0, 'alpha' : 0.5,
             'label' : 'Full Model Fit' if add_legend else None})
        plot_spectrum(fm.freqs, 10**fm.fooofed_spectrum_, log_freqs, log_powers,
                      ax=ax, plot_style=None, **model_kwargs)

        # Plot the aperiodic component of the model fit
        if plot_aperiodic:
            aperiodic_kwargs = check_plot_kwargs(aperiodic_kwargs, \
                {'color' : PLT_COLORS['aperiodic'], 'linewidth' : 3.0, 'alpha' : 0.5,
                 'linestyle' : 'dashed', 'label' : 'Aperiodic Fit' if add_legend else None})
            plot_spectrum(fm.freqs, 10**fm._ap_fit, log_freqs, log_powers,
                          ax=ax, plot_style=None, **aperiodic_kwargs)

    # Apply style to plot
    check_n_style(plot_style, ax, log_freqs, True)

    # Save out figure, if requested
    if save_fig:
        if not file_name:
            raise ValueError("Input 'file_name' is required to save out the plot.")
        plt.savefig(fpath(file_path, fname(file_name, 'png')))
