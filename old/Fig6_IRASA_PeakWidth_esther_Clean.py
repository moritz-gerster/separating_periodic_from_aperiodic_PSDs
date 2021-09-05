# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.signal as sig
from fooof import FOOOF
from helper import irasa

try:
    from tqdm import trange
except ImportError:
    trange = range

from functions import annotate_range6

# %% Plot parameters

# Save Path
fig_path = "../paper_figures/"
fig_name = "Fig6_PeakWidth_Esther.pdf"
Path(fig_path).mkdir(parents=True, exist_ok=True)

# File names
path = "../data/Fig5/"

# Sizes
lw = 2

# Colors
# a)
c_sim = "k"
c_error = "r"
c_noise = "darkgray"
c_ap = "grey"
c_range1 = "b"
c_range2 = "g"
c_range3 = "y"

# b)
c_real = "purple"

# c)
c_fooof = "deepskyblue"
c_IRASA1 = "C1"
c_IRASA2 = "C2"

# %% Real data

sample_rate = 2400
nperseg = sample_rate
welch_params = dict(fs=sample_rate, nperseg=nperseg)

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


filt = (freq <= 1200)
freq_filt = freq[filt]
spec_GRAD = spec_MEG[0, filt]
spec_MAG = spec_MEG[1, filt]


plot_small = (freq_filt, spec_GRAD, c_real)
plot_med = (freq_filt, spec_MAG, c_real)
plot_large = (freq, spec_LFP, c_real)

plot_med_low = (freq_filt, spec_MAG/10, c_real)
plot_large_low = (freq, spec_LFP/10, c_real)

# %% Apply fooof to determine peak width

real_s = FOOOF(max_n_peaks=2, verbose=False)
real_m = FOOOF(max_n_peaks=1, verbose=False, peak_width_limits=(0.5, 150))
real_l = FOOOF(max_n_peaks=1, verbose=False, peak_width_limits=(.5, 150))

real_s.fit(freq, spec_MEG[0], (3, 20))
real_m.fit(freq, spec_MEG[1], (1, 100))
real_l.fit(freq, spec_LFP, (10, 150))

freq_real_s, pow_real_s, bw_real_s = real_s.peak_params_[0]
freq_real_m, pow_real_m, bw_real_m = real_m.peak_params_[0]
freq_real_l, pow_real_l, bw_real_l = real_l.peak_params_[0]

# %% C Calc IRASA
h_max1 = 2
h_max2 = 25

# doesn't matter that shorter band makes more sence, this is topic of fig4
band = (1, 100)

band_h1 = (highpass * h_max1, lowpass / h_max1)
band_h2 = (highpass * h_max2, lowpass / h_max2)

N_h = 16

irasa_params1 = dict(sf=sample_rate, band=band_h1,
                     hset=np.linspace(1.1, h_max1, N_h))
irasa_params2 = dict(sf=sample_rate, band=band_h2,
                     hset=np.linspace(1.1, h_max2, N_h))

IRASA_small_h1 = irasa(MEG_raw[0], **irasa_params1)
IRASA_med_h1 = irasa(MEG_raw[1], **irasa_params1)
IRASA_large_h1 = irasa(LFP_raw, **irasa_params1)
IRASA_m_h2 = irasa(MEG_raw[1], **irasa_params2)
IRASA_l_h2 = irasa(LFP_raw, **irasa_params2)



freq_I_h1, ap_small_h1, per_small_h1, params_small_h1 = IRASA_small_h1
_, ap_med_h1, per_med_h1, params_med_h1 = IRASA_med_h1
_, ap_large_h1, per_large_h1, params_large_h1 = IRASA_large_h1
freq_I_h2, ap_med_h2, per_med_h2, params_med_h2 = IRASA_m_h2
freq_I_h2, ap_large_h2, per_large_h2, params_large_h2 = IRASA_l_h2

plot_ap_small_h1 = (freq_I_h1, ap_small_h1[0], c_IRASA1)
plot_ap_med_h1 = (freq_I_h1, ap_med_h1[0], c_IRASA1)
plot_ap_large_h1 = (freq_I_h1, ap_large_h1[0], c_IRASA1)
plot_ap_med_h2 = (freq_I_h2, ap_med_h2[0]/10, c_IRASA2)
plot_ap_large_h2 = (freq_I_h2, ap_large_h2[0]/10, c_IRASA2)

# Show what happens for larger freq ranges
irasa_params2["band"] = band_h1

IRASA_med_h1_long = irasa(MEG_raw[1], **irasa_params1)
IRASA_large_h1_long = irasa(LFP_raw, **irasa_params1)
IRASA_m_h2_long = irasa(MEG_raw[1], **irasa_params2)
IRASA_l_h2_long = irasa(LFP_raw, **irasa_params2)

(freq_I_long, ap_med_h1_long,
 per_med_h1_long, params_med_h1_long) = IRASA_med_h1_long
(_, ap_large_h1_long,
 per_large_h1_long, params_large_h1_long) = IRASA_large_h1_long
_, ap_med_h2_long, per_med_h2_long, params_med_h2_long = IRASA_m_h2_long
(_, ap_large_h2_long,
 per_large_h2_long, params_large_h2_long) = IRASA_l_h2_long

plot_ap_med_h1_long = (freq_I_long, ap_med_h1_long[0], c_IRASA1)
plot_ap_large_h1_long = (freq_I_long, ap_large_h1_long[0], c_IRASA1)
plot_ap_med_h2_long = (freq_I_long, ap_med_h2_long[0]/10, c_IRASA2)
plot_ap_large_h2_long = (freq_I_long, ap_large_h2_long[0]/10, c_IRASA2)


# %% Plot Settings

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

# %% Plot
fig, axes = plt.subplots(1, 3, figsize=[width, 2.7])

alpha_long = .3

# c1
ax = axes[0]
ax.loglog(*plot_small, label="Grad")
ax.loglog(*plot_ap_small_h1, label=r"$h_{max}$ = "f"{h_max1}")
ax.set_ylabel(ylabel_d)
ax.set_xlabel(xlabel)
ymin, ymax = ax.get_ylim()
ax.set_ylim((3e-26, ymax))
ax.text(s="a", **abc, transform=ax.transAxes)
# annotate freq bandwidth
xmin = freq_real_s - bw_real_s
xmax = freq_real_s + bw_real_s
ylow = plot_small[1][np.argmin(np.abs(plot_small[0] - xmin))]
yhigh = plot_small[1][np.argmin(np.abs(plot_small[0] - xmax))]
height = 6e-26
annotate_range6(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                height=height, annotate_pos="left")
ax.set_xticks(xticks_c)
ax.set_xticklabels(xticks_c)
ax.legend(loc=1, borderaxespad=0)


# c2
ax = axes[1]
ax.loglog(*plot_med, label="Mag")
ax.loglog(*plot_ap_med_h1)  # label=r"$h_{max}$ = "f"{h_max1}")
ax.loglog(*plot_ap_med_h1_long, ls="--", alpha=alpha_long)

ax.loglog(*plot_med_low, alpha=.5)
ax.loglog(*plot_ap_med_h2, label=r"$h_{max}$ = "f"{h_max2}")
ax.loglog(*plot_ap_med_h2_long, ls="--", alpha=alpha_long)
ax.set_ylabel(ylabel_e)
ax.set_xlabel(xlabel)
ax.text(s="b", **abc, transform=ax.transAxes)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)
ylim = (3e-27, 5e-23)
ax.set_ylim(ylim)
ax.legend(loc=1, borderaxespad=0)
# annotate freq bandwidth
xmin = freq_real_m - bw_real_m + 2
xmax = freq_real_m + bw_real_m + 2
ylow = plot_med_low[1][np.argmin(np.abs(plot_med_low[0] - xmin))]
yhigh = plot_med_low[1][np.argmin(np.abs(plot_med_low[0] - xmax))]
height = 5 * ylim[0]
annotate_range6(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                height=height, annotate_pos=.4)
ax.set_xticks(xticks_b)
ax.set_xticklabels(xticks_b)

# c3
ax = axes[2]
ax.loglog(*plot_large, label="LFP")
ax.loglog(*plot_ap_large_h1)  # label=r"$h_{max}$ = "f"{h_max1}")
ax.loglog(*plot_ap_large_h1_long, ls="--", alpha=alpha_long)

ax.loglog(*plot_large_low, alpha=.5)
ax.loglog(*plot_ap_large_h2)  # label=r"$h_{max}$ = "f"{h_max2}")
ax.loglog(*plot_ap_large_h2_long, ls="--", alpha=alpha_long)

ax.set_ylabel(ylabel_f)
ax.set_xlabel(xlabel)
ax.legend(loc=1, borderaxespad=0)
ylim = (1e-5, 2)
ax.set_ylim(ylim)
ax.text(s="c", **abc, transform=ax.transAxes)
x_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.xaxis.set_minor_locator(x_minor)
y_minor = mpl.ticker.LogLocator(subs=np.arange(0, 1, 0.1), numticks=10)
ax.yaxis.set_minor_locator(y_minor)
ax.set_yticklabels([], minor=True)

# annotate freq bandwidth
xmin = freq_real_l - bw_real_l
xmax = freq_real_l + bw_real_l
ylow = plot_large_low[1][np.argmin(np.abs(plot_large_low[0] - xmin))]
yhigh = plot_large_low[1][np.argmin(np.abs(plot_large_low[0] - xmax))]
height = 10 * ylim[0]
annotate_range6(ax, xmin=xmin, xmax=xmax, ylow=ylow, yhigh=yhigh,
                height=height, annotate_pos="left")
ax.set_xticks(xticks_b)
ax.set_xticklabels(xticks_b)

plt.tight_layout()
plt.savefig(fig_path + fig_name[:-4] + "SuppMat.pdf", bbox_inches="tight")
plt.show()

# %%
