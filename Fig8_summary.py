# %%
import numpy as np
import matplotlib.pyplot as plt
from fooof import FOOOF
import mne
from mne.time_frequency import psd_welch
from fooof.sim.gen import gen_aperiodic
from helper import irasa
supp = False


def annotate_range(ax, xmin, xmax, height, ylow=None, yhigh=None,
                   annotate_pos=True, annotation="log-diff",
                   annotation_fontsize=7):
    """
    Annotate fitting range or peak width.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Ax to draw the lines.
    xmin : float
        X-range minimum.
    xmax : float
        X-range maximum.
    height : float
        Position on y-axis of range.
    ylow : float, optional
        Position on y-axis to connect the vertical lines. If None, no vertical
        lines are drawn. The default is None.
    yhigh : float, optional
        Position on y-axis to connect the vertical lines. If None, no vertical
        lines are drawn. The default is None.
    annotate_pos : bool, optional
        Where to annotate.
    annotate : bool, optional
        The kind of annotation.
        "diff": Print range.
        "log-diff": Print range and logrange.
        else: Print range1-range2

    Returns
    -------
    None.

    """
    text_pos = 10**((np.log10(xmin) + np.log10(xmax)) / 2)
    box_alpha = 1
    ha = "center"
    text_height = height
    if annotate_pos == "below":
        text_height = 1.5e-1 * height
        box_alpha = 0
    elif isinstance(annotate_pos, (int, float)):
        text_height *= annotate_pos
        box_alpha = 0
    elif annotate_pos == "left":
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

    if annotation == "diff":
        range_str = f"{xmax-xmin:.0f} Hz"
    elif annotation == "log-diff":
        xdiff = xmax - xmin
        if xdiff > 50:  # round large intervals
            xdiff = np.round(xdiff, -1)
        range_str = (r"$\Delta f=$"f"{xdiff:.0f} Hz\n"
                     r"$\Delta f_{log}=$"f"{(np.log10(xmax/xmin)):.1f}")
    else:
        range_str = f"{xmin:.1f}-{xmax:.1f} Hz"
    ax.text(text_pos, text_height, s=range_str, **anno_dic)

    if ylow and yhigh:
        ax.vlines(xmin, height, ylow, **vline_dic)
        ax.vlines(xmax, height, yhigh, **vline_dic)


def detect_noise_floor(freq, psd, f_start, f_range=50, thresh=0.05,
                       step=1, reverse=False,
                       ff_kwargs=dict(verbose=False, max_n_peaks=1)):
    """
    Detect the plateau of a power spectrum where the slope a < thresh.

    Parameters
    ----------
    freq : ndarray
        Freq array.
    psd : ndarray
        PSD array.
    f_start : float
        Starting frequency for the search.
    f_range : int, optional
        Fitting range.
        If set low, more susceptibility to noise/peaks.
        If set large, less spatial precision.
        The default is 50.
    thresh : float, optional
        Threshold for plateau. The default is 0.05.
    step : int, optional
        Step of loop over fitting range. The default is 1 which might take
        unneccessarily long computation time for maximum precision.
    reverse : bool, optional
        If True, start at high frequencies and detect the end of a pleateau.
        The default is False.
    ff_kwargs : dict, optional
        Fooof fitting keywords.
        The default is dict(verbose=False, max_n_peaks=1). There shouldn't be
        peaks close to the plateau but fitting at least one peak is a good
        idea for power line noise.

    Returns
    -------
    n_start : int
        Start frequency of plateau.
        If reverse=True, end frequency of plateau.
    """
    exp = 1
    fm = FOOOF(**ff_kwargs)
    while exp > thresh:
        if reverse:
            f_start -= step
            freq_range = [f_start - f_range, f_start]
        else:
            f_start += step
            freq_range = [f_start, f_start + f_range]
        fm.fit(freq, psd, freq_range)
        exp = fm.get_params('aperiodic_params', 'exponent')
    return f_start + f_range // 2


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
fig_name = "Fig8_Summary_anno"

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

spec5_real = freq, spec5, c_empirical
spec9_real = freq, spec9, c_empirical

spec5_fooof = fm5.freqs, 10**fm5_fooof, c_fooof
spec9_fooof = fm9.freqs, 10**fm9_fooof, c_fooof

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


fig, ax = plt.subplots(2, 2, figsize=(fig_width, 5), sharey="row")

ax[0, 0].text(s='    "Easy" spectrum', **panel_description,
              transform=ax[0, 0].transAxes)
ax[1, 0].text(s='    "Hard" spectrum', **panel_description,
              transform=ax[1, 0].transAxes)
# lin
ax[0, 0].semilogy(*spec5_real, label="Sub 5 MEG")  # + ch5)
ax[1, 0].semilogy(*spec9_real, label="Sub 9 LFP")  # + ch9)

# log
ax[0, 1].loglog(*spec5_real, label="Sub 5 MEG")
ax[1, 1].loglog(*spec9_real, label="Sub 9 LFP")

# Fooof fit
ax[0, 1].loglog(*spec5_fooof, label=f"fooof     a={a5_fooof:.2f}")
ax[1, 1].loglog(*spec9_fooof, label=f"fooof     a={a9_fooof:.2f}")

# Straight fit
ax[0, 1].loglog(*spec5_straight, label=f"straight a={a5_straight:.2f}")
ax[1, 1].loglog(*spec9_straight, label=f"straight a={a9_straight:.2f}")

# Low fit
ax[0, 1].loglog(*spec5_I, label=f"IRASA    a={a5_I:.2f}")
ax[1, 1].loglog(*spec9_I, label=f"IRASA    a={a9_I:.2f}")

for axes in ax.flatten():
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    # axes.legend(loc=1)

# Legend
handles, labels = ax[0, 1].get_legend_handles_labels()
ax[0, 0].legend(handles, labels)
handles, labels = ax[1, 1].get_legend_handles_labels()
ax[1, 0].legend(handles, labels)

# Add Plateau rectangle
ylim_b = (5e-3, 6)
xlim_b = ax[1, 0].get_xlim()
noise_start = detect_noise_floor(freq, spec9, 50)
rec_xy = (noise_start, ylim_b[0])
rec_width = freq[-1] - noise_start
rec_height = np.diff(ylim_b)[0]
rect_c = dict(xy=rec_xy, width=rec_width, height=rec_height,
              alpha=.15, color="r")
# ax[1, 0].add_patch(plt.Rectangle(**rect_c))
ax[1, 1].add_patch(plt.Rectangle(**rect_c))

# Add Plateau annotation
# ax[1, 0].hlines(spec9[noise_start], noise_start, freq[-1], color="k",
#                linewidth=1)
ax[1, 1].hlines(spec9[noise_start], noise_start, freq[-1], color="k",
                linewidth=1)
ax[1, 1].annotate(s="Early\nPlateau\nonset",
                  xy=(noise_start, spec9[noise_start]),
                  xytext=(noise_start, spec9[noise_start]*20),
                  arrowprops=dict(arrowstyle="->", shrinkB=5),
                  color="k", fontsize=8,
                  ha="left",
                  verticalalignment="center")

# Add Peak width annotation
height1 = 100
# height2 = 20
# height3 = 10
xmin1 = freq5_1 - bw5_1
xmax1 = freq5_1 + bw5_1
# xmin2 = freq5_2 - bw5_2
# xmax2 = freq5_2 + bw5_2
# xmin3 = freq5_3 - bw5_3
# xmax3 = freq5_3 + bw5_3
annotate_range(ax[0, 1], xmin1, xmax1, height1, annotate_pos="left")
# annotate_range(ax[0, 1], xmin2, xmax2, height2, annotate_pos="left")
# annotate_range(ax[0, 1], xmin3, xmax3, height3, annotate_pos="left")

# Add Peak width annotation
height1 = .029
height2 = 0.009
xmin1 = freq9_1 - bw9_1
xmax1 = freq9_1 + bw9_1
xmin2 = freq9_2 - bw9_2
xmax2 = freq9_2 + bw9_2
annotate_range(ax[1, 1], xmin1, xmax1, height1, annotate_pos=.93)
annotate_range(ax[1, 1], xmin2, xmax2, height2, annotate_pos=.93)

# Add overlaps
# 2. Peak stimmt nicht
# def gauss(fm: FOOOF, n_peak: int) -> np.ndarray:
#     """Plot Gaussian peaks fitted with fooof.
#     Args:
#         mean (float): center frequency
#         height (float): height
#         std (float): standard deviation
#         freq (nd.array): Freq array
#     Returns:
#         nd.array: Gaussian function for plotting.
#     """
#     mean, height, std = fm9.gaussian_params_[n_peak]
#     gauss_fit = height * np.exp(-(fm.freqs-mean)**2 / (2 * std**2))
#     aperiodic = 10**gen_aperiodic(fm.freqs, fm.aperiodic_params_)
#     return aperiodic + gauss_fit
# gauss_fit1 = gauss(fm9, 0)
# gauss_fit2 = gauss(fm9, 1)
# ax[1, 1].plot(fm9.freqs, gauss_fit1)
# ax[1, 1].plot(fm9.freqs, gauss_fit2)
# ax[1, 1].plot(fm9.freqs, fm9.fooofed_spectrum_)

# Add indication of peak overlap as vertical arrow
overlap = 15
arr_height = 1
ax[1, 1].annotate(s="", xy=(overlap, spec9[overlap]),
                  xytext=(overlap, 10**fm9_straight[overlap]),
                  arrowprops=dict(arrowstyle="<->"))
ax[1, 1].annotate(s="", xy=(freq9_1, arr_height),
                  xytext=(freq9_2, arr_height),
                  arrowprops=dict(arrowstyle="<->"))
ax[1, 1].text(s="Broad\nPeak\nWidths:", x=1, y=(height1+height2)/2, ha="left",
              va="center", fontsize=8)
ax[1, 1].text(s="Peak\nOverlap", x=overlap, y=arr_height*.9, ha="left",
              va="top", fontsize=8)

# Annotate orders of magnitude
ord_magn5 = int(np.round(np.log10(spec5[0]/spec5[-1])))
x_line = -25
ax[0, 0].annotate(s="",
                  xy=(x_line, spec5[0]),
                  xytext=(x_line, spec5[-1]),
                  arrowprops=dict(arrowstyle="|-|,widthA=.5,widthB=.5",
                                  lw=1.3),
                  ha="center")
ax[0, 0].text(s=rf"$\Delta PSD\approx 10^{{{ord_magn5}}}$", x=30,
              y=np.sqrt(spec5[0]*spec5[-1]), va="center", fontsize=8)

ord_magn9 = int(np.round(np.log10(spec9[0]/spec9[-1])))
x_line = -25
ax[1, 0].annotate(s="",
                  xy=(x_line, spec9[0]),
                  xytext=(x_line, spec9[-1]),
                  arrowprops=dict(arrowstyle="|-|,widthA=.5,widthB=.5",
                                  lw=1.3), ha="center")
ax[1, 0].text(s=rf"$\Delta PSD\approx 10^{{{ord_magn9}}}$", x=55,
              y=np.sqrt(spec9[0]*spec9[-1]), va="center", fontsize=8)

xlim5 = ax[0, 0].get_xlim()
xlim9 = ax[1, 0].get_xlim()
ax[0, 0].set(xlabel=None, ylabel="A.U. Voxel Data", xlim=(-50, xlim5[1]))
ax[1, 0].set(xlabel=None, ylabel=None, xlim=(-50, xlim9[1]))
ax[1, 0].set(xlabel="Frequency [Hz]", ylabel=r"PSD [$\mu$$V^2/Hz$]")
ax[1, 1].set(xlabel="Frequency [Hz]", ylabel=None, ylim=ylim_b)
ax[0, 0].text(s="a", **panel_labels, transform=ax[0, 0].transAxes)
ax[1, 0].text(s="b", **panel_labels, transform=ax[1, 0].transAxes)

plt.tight_layout()
plt.savefig(fig_path + fig_name + ".pdf", bbox_inches="tight")
plt.savefig(fig_path + fig_name + ".png", dpi=1000, bbox_inches="tight")
plt.show()
