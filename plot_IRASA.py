"""Plot IRASA."""
import os
import B_config
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from mne.time_frequency import psd_welch
from C_fooof import FOOOF
from pathlib import Path
import C_fooof
import yasa
import seaborn as sns

ch_names = B_config.ch_names
conditions = B_config.conditions
n_sub = B_config.n_sub
n_ch = B_config.n_ch

filter_params = B_config.filter_params
slope_ranges = B_config.slope_ranges
welch_params = B_config.welch_params
fooof_params = B_config.fooof_params

coh_bands = B_config.coh_bands


raw_on, raw_off = C_fooof.load_fif()
freqs, (spect_on, spect_off) = C_fooof.MNE_calc_spectra(raw_on, raw_off)

ch = 0
subj = 0
raw_on_subj = raw_on[subj]
data_on_subj = raw_on_subj.get_data()
sfreq = B_config.sfreq
win_sec = 0.5
ch_names = B_config.ch_names

freq_range = [1, 95]

# %%

# Apply the IRASA technique
IRASA = yasa.irasa(data_on_subj, sfreq, ch_names=ch_names, band=freq_range,
                   win_sec=win_sec, kwargs_welch={'average': 'mean'})

freqs_i, psd_aperiodic_i, psd_osc_i, fit_params_i = IRASA

assert np.all(freqs == freqs_i)

# %%

fig, ax = plt.subplots(1, 2, figsize=[10, 5], sharey=False, sharex=True)

ax[0].plot(freqs_i, psd_aperiodic_i[ch, :],
           "b", linestyle="--", lw=2, label="aperiodic")
ax[0].plot(freqs_i, psd_osc_i[ch, :],
           "grey", lw=2, label="oscillatory")
ax[0].plot(freqs_i, psd_osc_i[ch, :] + psd_aperiodic_i[ch, :],
           "k", lw=2, label="sum")
ax[0].plot(freqs, spect_on[subj][ch], "g", label="orig")
ax[0].set_xlim([1, 100])
ax[0].set_yscale("log")
sns.despine(ax=ax[0])
ax[0].set_title(ch_names[ch])
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_ylabel('PSD log($uV^2$/Hz)')
ax[0].legend()

fm = FOOOF(**B_config.fooof_params)
fm.fit(freqs, spect_on[subj][ch], freq_range)
fm.plot(ax=ax[1])
ax[1].set_title(ch_names[ch])

plt.tight_layout()
plt.show()

slope_f = -fm.get_params('aperiodic_params', 'exponent')
offset_f = fm.get_params('aperiodic_params', 'offset')
slope_i = fit_params_i["Slope"].iloc[ch]
offset_i = fit_params_i["Intercept"].iloc[ch]

