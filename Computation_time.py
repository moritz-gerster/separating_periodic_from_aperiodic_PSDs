# %%
import mne
import numpy as np
import scipy.signal as sig
from fooof import FOOOFGroup

from utils import irasa

# File names
path = "../data/Fig5/"
fname = "subj6_off_R1_raw.fif"
sub = mne.io.read_raw_fif(path + fname, preload=True)

# Convert mne to numpy
srate = 2400
start = int(0.5 * srate)  # artefact in beginning of recording
stop = int(185 * srate)  # artefact at the end of recording

sub = sub.get_data(start=start, stop=stop)[:9]

band = (1, 30)

# %% IRASA at default parameters
%%timeit
# =============================================================================
# IRASA time = 12.1 +- 0.2 seconds
# =============================================================================

IRASA = irasa(data=sub, sf=srate, band=band)
_, _, _, params = IRASA

# %% IRASA does not slow down if spectral resolution is increased/decreased
%%timeit
# =============================================================================
# IRASA time = 12.4 +- 0.3
# =============================================================================

IRASA = irasa(data=sub, sf=srate, band=band, win_sec=20)
_, _, _, params = IRASA

# %% IRASA does not slow down if spectral resolution is increased/decreased
%%timeit
# =============================================================================
# IRASA time = 12 +- 0.4
# =============================================================================

IRASA = irasa(data=sub, sf=srate, band=band, win_sec=0.5)
_, _, _, params = IRASA

# %% IRASA slows down if number of hset is increased
%%timeit
# =============================================================================
# IRASA time = 55 +- 0.2
# =============================================================================

IRASA = irasa(data=sub, sf=srate, band=band, hset=np.arange(1.1, 1.9, 0.01))
_, _, _, params = IRASA

# %% IRASA slows down if values of hset is increased
%%timeit
# =============================================================================
# IRASA time = 1min 6s ± 2.02 s per loop (mean ± std. dev. of 7 runs,
# 1 loop each)
# =============================================================================

IRASA = irasa(data=sub, sf=srate, band=band, hset=np.arange(10.1, 10.9, 0.05))
_, _, _, params = IRASA

# %% Fooof fast even if psd calc included
%%timeit
# =============================================================================
# FOOOF 265 ms +- 5 ms
# =============================================================================

# Signal params
welch_params = dict(fs=srate, nperseg=srate)
freq, psd_sub = sig.welch(sub, **welch_params)

fm = FOOOFGroup(verbose=False)
fm.fit(freq, psd_sub, band)

# %% Fooof twice as fast without PSD calculcation
%%timeit
# =============================================================================
# FOOOF 123 ms +- 1.1 ms
# =============================================================================

fm = FOOOFGroup(verbose=False)
fm.fit(freq, psd_sub, band)

# %% Fooof slow if spectral resolution too high
%%timeit
# =============================================================================
# FOOOF 409 ms ± 3.96 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# =============================================================================

# Signal params
welch_params = dict(fs=srate, nperseg=4*srate)
freq, psd_sub = sig.welch(sub, **welch_params)

fm = FOOOFGroup(verbose=False)
fm.fit(freq, psd_sub, band)

# %%
