# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from mne.time_frequency import psd_welch

from utils import irasa

supp = False
# %% Parameters b)

srate_pd = 2400

# Colors
c_empirical = "purple"
c_straight = "r--"
c_fooof = "b--"
c_low = "g--"


# Paths
data_path = "../data/Fig7/"
fig_path = "../paper_figures/"
fig_name = "Fig_Poster"

fooof_params = dict(verbose=False, peak_width_limits=(0.5, 150))

# %% Get data b)

sub5 = mne.io.read_raw_fif(data_path + "subj5_on_R1_raw.fif", preload=True)
sub9 = mne.io.read_raw_fif(data_path + "subj9_on_R8_raw.fif", preload=True)

ch5 = "SMA"
ch9 = "STN_R01"
sub5.pick_channels([ch5])
sub9.pick_channels([ch9])
filter_params = {"freqs": np.arange(50, 601, 50),
                 "notch_widths": 0.1,
                 "method": "spectrum_fit"}
sub5.notch_filter(**filter_params)
sub9.notch_filter(**filter_params)

welch_params_b = {"fmin": 1,
                  "fmax": 600,
                  "tmin": 0.5,
                  "tmax": 185,
                  "n_fft": srate_pd,
                  "n_overlap": srate_pd // 2,
                  "average": "mean"}

spec5, freq = psd_welch(sub5, **welch_params_b)
spec9, freq = psd_welch(sub9, **welch_params_b)

spec5 = spec5[0]
spec9 = spec9[0]

# %% Fit b)
freq_range = [1, 95]

fm5 = FOOOF(**fooof_params)
fm9 = FOOOF(**fooof_params)

fm5.fit(freq, spec5, freq_range)
fm9.fit(freq, spec9, freq_range)

# Fooof fit
fm5_fooof = gen_aperiodic(fm5.freqs, fm5.aperiodic_params_)
fm9_fooof = gen_aperiodic(fm9.freqs, fm9.aperiodic_params_)

a5_fooof = fm5.aperiodic_params_[1]
a9_fooof = fm9.aperiodic_params_[1]

# Get peak params
freq5_1, pow5_1, bw5_1 = fm5.peak_params_[1]
freq5_2, pow5_2, bw5_2 = fm5.peak_params_[2]
freq5_3, pow5_3, bw5_3 = fm5.peak_params_[3]

freq9_1, pow9_1, bw9_1 = fm9.peak_params_[0]
freq9_2, pow9_2, bw9_2 = fm9.peak_params_[1]

# Straight fit
DeltaX = np.log10(np.diff(freq_range)[0])

offset5 = np.log10(spec5[freq == freq_range[0]][0])
endpoint5 = np.log10(spec5[freq == freq_range[1]][0])
DeltaY5 = offset5 - endpoint5

offset9 = np.log10(spec9[freq == freq_range[0]][0])
endpoint9 = np.log10(spec9[freq == freq_range[1]][0])
DeltaY9 = offset9 - endpoint9

a5_straight = DeltaY5 / DeltaX
a9_straight = DeltaY9 / DeltaX

fm5_straight = gen_aperiodic(fm5.freqs, np.array([offset5, a5_straight]))
fm9_straight = gen_aperiodic(fm9.freqs, np.array([offset9, a9_straight]))

# IRASA fit
get_data = dict(start=srate_pd//2, stop=srate_pd*180,
                reject_by_annotation="NaN")
sub5_I = sub5.get_data(**get_data)
sub9_I = sub9.get_data(**get_data)

freq_I, _, _, params5_I = irasa(sub5_I, band=freq_range, sf=srate_pd)
_, _, _, params9_I = irasa(sub9_I, band=freq_range, sf=srate_pd)

off5_I = params5_I["Intercept"][0]
off9_I = params9_I["Intercept"][0]

a5_I = -params5_I["Slope"][0]
a9_I = -params9_I["Slope"][0]

fm5_I = gen_aperiodic(freq_I, np.array([off5_I, a5_I]))
fm9_I = gen_aperiodic(freq_I, np.array([off9_I, a9_I]))

# Low fit
# =============================================================================
# offset5_low = np.log10(spec5[freq == freq_range[0]][0] * 0.5)
# DeltaY5_low = offset5_low - endpoint5
#
# offset9_low = np.log10(spec9[freq == freq_range[0]][0] * 0.5)
# DeltaY9_low = offset9_low - endpoint9
#
# a5_low = DeltaY5_low / DeltaX
# a9_low = DeltaY9_low / DeltaX
#
# fm5_low = gen_aperiodic(fm5.freqs, np.array([offset5_low, a5_low]))
# fm9_low = gen_aperiodic(fm9.freqs, np.array([offset9_low, a9_low]))
# =============================================================================

spec5_real = freq, spec5, "#001EC4"
spec9_real = freq, spec9, "#001EC4"

spec5_fooof = fm5.freqs, 10**fm5_fooof, "#FF8A00"
spec9_fooof = fm9.freqs, 10**fm9_fooof, "#FF8A00"

spec5_straight = fm5.freqs, 10**fm5_straight, c_straight
spec9_straight = fm9.freqs, 10**fm9_straight, c_straight

spec5_I = freq_I, 10**fm5_I, c_low
spec9_I = freq_I, 10**fm9_I, c_low

# =============================================================================
# spec5_low = fm5.freqs, 10**fm5_low, c_low
# spec9_low = fm9.freqs, 10**fm9_low, c_low
# =============================================================================
# %% Fig Params

fig_width = 6.85  # inches
panel_fontsize = 12
panel_labels = dict(x=0, y=1.02, fontsize=panel_fontsize,
                    fontdict=dict(fontweight="bold"))
panel_description = dict(x=0, y=1.02, fontsize=panel_fontsize)

# %% Plot

fig, ax = plt.subplots(1, 2, figsize=(fig_width, 3))


# log
ax[0].loglog(*spec5_real, label="MEG")
ax[1].loglog(*spec9_real, label="LFP")

# Fooof fit
ax[0].loglog(*spec5_fooof, label=rf"$\beta$={a5_fooof:.2f}")
ax[1].loglog(*spec9_fooof, label=rf"$\beta$={a9_fooof:.2f}")
ax[0].axes.get_xaxis().set_visible(False)
ax[1].axes.get_xaxis().set_visible(False)
ax[0].axes.get_yaxis().set_visible(False)
ax[1].axes.get_yaxis().set_visible(False)
# ax[0].axis("off")
# ax[1].axis("off")

for axes in ax.flatten():
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    # axes.legend(loc=1)

# Legend
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles, labels)
handles, labels = ax[1].get_legend_handles_labels()
ax[1].legend(handles, labels)

xlim5 = ax[0].get_xlim()
xlim9 = ax[1].get_xlim()
ax[0].set(xlabel="Frequency [Hz]",
          ylabel="PSD", xlim=(-50, xlim5[1]))
ax[1].set(xlabel="Frequency [Hz]",
          ylabel=None, xlim=(-50, xlim9[1]))
# ax[0].text(s="a", **panel_labels, transform=ax[0].transAxes)
# ax[1].text(s="b", **panel_labels, transform=ax[1].transAxes)

plt.tight_layout()
plt.savefig(fig_path + fig_name + ".pdf", bbox_inches="tight")
plt.savefig(fig_path + fig_name + ".png", dpi=1000, bbox_inches="tight")
plt.show()


# %%

# %%
