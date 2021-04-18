"""Compare fooof with IRASA on real data."""
import numpy as np
# import scipy.signal as sig
from scipy.stats import norm
import matplotlib.pyplot as plt
# import matplotlib.ticker
from fooof import FOOOF, FOOOFGroup
from fooof.sim.gen import gen_aperiodic
# from pathlib import Path
import pandas as pd
# import mne
from helper import load_fif, load_psd, irasa

params = {'legend.fontsize': 12.2,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'xtick.minor.size': 0,
          'xtick.major.size': 5,
          'ytick.major.size': 5,
          'lines.linewidth': 2}
plt.rcParams.update(params)




# %% PARAMETERS

# Signal
srate = 2400
time = np.arange(180 * srate)
samples = time.size
slopes = np.arange(0, 4.5, .5)
start = int(0.5 * srate)
stop = int(185 * srate)

# WELCH
win_sec = 1
nperseg = int(win_sec * srate)

# Path
fig_path = "plots/"

# Colors
c_on = "#E66101"
c_off = "#5E3C99"


# %% LOAD DATA
raw_conds = load_fif()
freqs, PSD_on, PSD_off = load_psd()

psds = (PSD_on, PSD_off)
# %% Select data
ch_names = ['SMA', 'leftM1', 'rightM1',
            'STN_R01', 'STN_R12', 'STN_R23',
            'STN_L01', 'STN_L12', 'STN_L23']

conds = ["on", "off"]
# (subj, ch, cond)
easy = [(4, 0, 0), (4, 1, 0), (4, 5, 0), (0, 3, 0), (1, 8, 0), (6, 0, 0),
        (6, 5, 0), (7, 5, 0), (7, 6, 1), (9, 5, 0), (9, 6, 0), (10, 5, 0)]

hard = [(0, 0, 0),  # broad
        (1, 3, 0),  # noise floor
        (1, 6, 0),  # many small peaks
        (2, 1, 0),  # broad
        (5, 2, 0),  # broad
        (5, 8, 0),  # unsual, big peaks
        (8, 3, 0), # huge, broad
        (11, 0, 0),  # broad
        (11, 2, 0)]  # broad


# =============================================================================
# for i in range(len(hard)):
# 
#     data = raw_conds[hard[i][2]][hard[i][0]]
#     signal_on = data.get_data(start=start, stop=stop,
#                               reject_by_annotation="nan")[hard[i][1]]
#     
#     
#     psd_np = psds[hard[i][2]][hard[i][0], hard[i][1]]
#     
#     freq_range = [0, 95]
#     
#     mask = (freqs > freq_range[0]) & (freqs < freq_range[1])
#     
#     fig, ax = plt.subplots(1, 2, figsize=[8, 5])
#     ax[0].semilogy(freqs[mask], psd_np[mask])
#     ax[1].loglog(freqs[mask], psd_np[mask])
#     subj = hard[i][0] + 1
#     ch = ch_names[hard[i][1]]
#     cond = conds[hard[i][2]]
#     fig.suptitle(f"Subj {subj}, {ch}, "
#                  f"{cond} hard")
#     plt.savefig(f"../plots/easy-hard/subj{subj}_{ch}_{cond}_hard.pdf")
#     plt.show()
#     
# for i in range(len(easy)):
# 
#     data = raw_conds[easy[i][2]][easy[i][0]]
#     signal_on = data.get_data(start=start, stop=stop,
#                               reject_by_annotation="nan")[easy[i][1]]
#     
#     
#     psd_np = psds[easy[i][2]][easy[i][0], easy[i][1]]
#     
#     freq_range = [0, 95]
#     
#     mask = (freqs > freq_range[0]) & (freqs < freq_range[1])
#     
#     fig, ax = plt.subplots(1, 2, figsize=[8, 5])
#     ax[0].semilogy(freqs[mask], psd_np[mask])
#     ax[1].loglog(freqs[mask], psd_np[mask])
#     fig.suptitle(f"Subj {easy[i][0]+1}, {ch_names[easy[i][1]]}, "
#                  f"{conds[easy[i][2]]} easy")
#     subj = easy[i][0] + 1
#     ch = ch_names[easy[i][1]]
#     cond = conds[easy[i][2]]
#     fig.suptitle(f"Subj {subj}, {ch}, "
#                  f"{cond} easy")
#     plt.savefig(f"../plots/easy-hard/subj{subj}_{ch}_{cond}_easy.pdf")
#     plt.show()
# =============================================================================





# %% Tune parameters easy


fig, axes = plt.subplots(2, 3, figsize=[20, 12])

j = 0
    
for i in range(9, 12):

    
    psd = psds[easy[i][2]][easy[i][0], easy[i][1]]
    data = raw_conds[easy[i][2]][easy[i][0]]
    signal = data.get_data(start=start, stop=stop,
                               reject_by_annotation="nan")[easy[i][1]]
    # % Tune Parameters Easy
    
    
    freq_range_f = [1, 45]
    freq_range_i = [10, 20]
    
    
    mask = (freqs >= freq_range_i[0]) & (freqs <= freq_range_i[1])
    
    fooof_params = {"peak_width_limits": (1, 12), "peak_threshold": 2,
                          "max_n_peaks": np.inf, "min_peak_height": .12,
                          "aperiodic_mode": "fixed"}
    irasa_params = {"band": freq_range_i,
                          "sf": srate, "win_sec": 1,
                          "hset": np.linspace(1.1, 9.9, num=15), # 2.9
                          "kwargs_welch": {'average': 'mean'}}

    # IRASA
    IRASA_real = irasa(data=signal, **irasa_params)
    freq, aperiodic, osc, params = IRASA_real

    i_sum = aperiodic[0] + osc[0]

    osc = osc[0]
    aperiodic = aperiodic[0]

    # Extract Fit Params
    offset = params["Intercept"].iloc[0]
    slope = -params["Slope"].iloc[0]

    # Calc fit
    fit = 10**offset / freq ** slope
    i_fit = (freq, fit, "b--") 
    # for different freq ranges the IRASA offsets get off

    # Extract error params
    r_squ = params["R^2"].iloc[0]

    # Calc mean error
    R2_mean_I = r_squ

    # Labels
    label = f"Real on 1/f {slope:.2f}"

    mask_full = (freqs <= 95)
    real_full = (freqs[mask_full], psd[mask_full])

    real = (freqs[mask], psd[mask])
    kwgs = {"c": "k", "label": "Original Spectrum"}

    # fooof
    fm = FOOOF(**fooof_params)  # Init fooof
    fm.fit(freqs, psd, freq_range_f)
    ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
    fooof_osc_orig = fm.power_spectrum - ap_fit
    fooof_osc = fm.fooofed_spectrum_ - ap_fit
    exponent = fm.get_params('aperiodic_params', 'exponent')
    r_squ = fm.get_params('r_squared')

    # IRASA OSc
    ir = (freq, i_sum, "r")
    ir_ap = (freq, aperiodic, "darkgreen")
    fit_orig = (10**offset / freqs[mask] ** slope)
    irasa_osc_orig = np.log10(psd[mask]) - np.log10(fit_orig)

    ax = axes[0, j]       
    freq_name = f"{freq_range_f[0]}-{freq_range_f[1]}Hz"
    subj = easy[i][0]+1
    ch = ch_names[easy[i][1]]
    cond = conds[easy[i][2]]
    title = f"Subj {subj} {ch} {cond} easy"
    ax.set_title(title, fontsize=25)

    ax.loglog(*real_full, **kwgs)
    ax.loglog(fm.freqs, 10**ap_fit, "brown", lw=3,
              label=f"fooof 1/f={exponent:.2f}")
    ax.loglog(*i_fit, lw=5, label=f"IRASA 1/f={slope:.2f}")
    ax.legend()

    ax = axes[1, j]
    ax.semilogy(*real_full, **kwgs)
    ax.semilogy(fm.freqs, 10**ap_fit, "brown", lw=3,
                label=f"fooof 1/f={exponent:.2f}")
    ax.semilogy(*i_fit, lw=5, label=f"IRASA 1/f={slope:.2f}")
    ax.legend()

    j += 1

plt.tight_layout()
plt.savefig(f"../plots/real/{freq_name}_IRASA_hset_max={irasa_params['hset'][-1]}_{subj}_{ch}_easy{i}.pdf")
plt.show()

























# %% Tune Parameters Hard

fig, axes = plt.subplots(2, 3, figsize=[20, 12])

j = 0


for i in range(5, 8):

    psd = psds[hard[i][2]][hard[i][0], hard[i][1]]
    data = raw_conds[hard[i][2]][hard[i][0]]
    signal = data.get_data(start=start, stop=stop,
                               reject_by_annotation="nan")[hard[i][1]]
    # % Tune Parameters Easy


    freq_range_f = [1, 45]
    freq_range_i = [10, 20]


    mask = (freqs >= freq_range_i[0]) & (freqs <= freq_range_i[1])

    fooof_params = {"peak_width_limits": (1, 12), "peak_threshold": 2,
                          "max_n_peaks": np.inf, "min_peak_height": .12,
                          "aperiodic_mode": "fixed"}
    irasa_params = {"band": freq_range_i,
                          "sf": srate, "win_sec": 1,
                          "hset": np.linspace(1.1, 9.9, num=15), # 2.9
                          "kwargs_welch": {'average': 'mean'}}

    # IRASA
    IRASA_real = irasa(data=signal, **irasa_params)
    freq, aperiodic, osc, params = IRASA_real

    i_sum = aperiodic[0] + osc[0]

    osc = osc[0]
    aperiodic = aperiodic[0]

    # Extract Fit Params
    offset = params["Intercept"].iloc[0]
    slope = -params["Slope"].iloc[0]

    # Calc fit
    fit = 10**offset / freq ** slope
    i_fit = (freq, fit, "b--") 
    # for different freq ranges the IRASA offsets get off

    # Extract error params
    r_squ = params["R^2"].iloc[0]

    # Calc mean error
    R2_mean_I = r_squ

    # Labels
    label = f"Real on 1/f {slope:.2f}"

    mask_full = (freqs <= 95)
    real_full = (freqs[mask_full], psd[mask_full])

    real = (freqs[mask], psd[mask])
    kwgs = {"c": "k", "label": "Original Spectrum"}

    # fooof
    fm = FOOOF(**fooof_params)  # Init fooof
    fm.fit(freqs, psd, freq_range_f)
    ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
    fooof_osc_orig = fm.power_spectrum - ap_fit
    fooof_osc = fm.fooofed_spectrum_ - ap_fit
    exponent = fm.get_params('aperiodic_params', 'exponent')
    r_squ = fm.get_params('r_squared')

    # IRASA OSc
    ir = (freq, i_sum, "r")
    ir_ap = (freq, aperiodic, "darkgreen")
    fit_orig = (10**offset / freqs[mask] ** slope)
    irasa_osc_orig = np.log10(psd[mask]) - np.log10(fit_orig)


    ax = axes[0, j]
    freq_name = f"{freq_range_f[0]}-{freq_range_f[1]}Hz"
    subj = hard[i][0]+1
    ch = ch_names[hard[i][1]]
    cond = conds[hard[i][2]]
    title = f"Subj {subj} {ch} {cond} hard"
    ax.set_title(title, fontsize=25)

    ax.loglog(*real_full, **kwgs)
    ax.loglog(fm.freqs, 10**ap_fit, "brown", lw=5, label=f"fooof 1/f={exponent:.2f}")
    ax.loglog(*i_fit, lw=5, label=f"IRASA 1/f={slope:.2f}")
    ax.legend()


    ax = axes[1, j]
    ax.semilogy(*real_full, **kwgs)
    ax.semilogy(fm.freqs, 10**ap_fit, "brown", lw=5, label=f"fooof 1/f={exponent:.2f}")
    ax.semilogy(*i_fit, lw=5, label=f"IRASA 1/f={slope:.2f}")
    ax.legend()


    j += 1

plt.tight_layout()
plt.savefig(f"../plots/real/{freq_name}_IRASA_hset_max={irasa_params['hset'][-1]}_{subj}_{ch}_hard{i}.pdf")
plt.show()



# %% Compare fooof IRASA

i=0

psd = psds[easy[i][2]][easy[i][0], easy[i][1]]
data = raw_conds[easy[i][2]][easy[i][0]]
signal = data.get_data(start=start, stop=stop,
                           reject_by_annotation="nan")[easy[i][1]]
# % Tune Parameters Easy


freq_range_f = [3, 95]
freq_range_i = [3, 95]


mask = (freqs >= freq_range_i[0]) & (freqs <= freq_range_i[1])

fooof_params = {"peak_width_limits": (1, 12), "peak_threshold": 2,
                      "max_n_peaks": np.inf, "min_peak_height": .12,
                      "aperiodic_mode": "fixed"}
irasa_params = {"band": freq_range_i,
                      "sf": srate, "win_sec": 1,
                      "hset": np.linspace(1.1, 2.9, num=15), # 2.9
                      "kwargs_welch": {'average': 'mean'}}




# % Calc and Plot

# IRASA
IRASA_real = irasa(data=signal, **irasa_params)
freq, aperiodic, osc, params = IRASA_real

i_sum = aperiodic[0] + osc[0]

osc = osc[0]
aperiodic = aperiodic[0]

# Extract Fit Params
offset = params["Intercept"].iloc[0]
slope = -params["Slope"].iloc[0]

# Calc fit
fit = 10**offset / freq ** slope
i_fit = (freq, fit, "b--") 
# for different freq ranges the IRASA offsets get off

# Extract error params
r_squ = params["R^2"].iloc[0]

# Calc mean error
R2_mean_I = r_squ

# Labels
label = f"Real on 1/f {slope:.2f}"

mask_full = (freqs <= 95)
real_full = (freqs[mask_full], psd[mask_full])

real = (freqs[mask], psd[mask])
kwgs = {"c": "k", "label": "Original Spectrum"}

# fooof
fm = FOOOF(**fooof_params)  # Init fooof
fm.fit(freqs, psd, freq_range_f)
ap_fit = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
fooof_osc_orig = fm.power_spectrum - ap_fit
fooof_osc = fm.fooofed_spectrum_ - ap_fit
exponent = fm.get_params('aperiodic_params', 'exponent')
r_squ = fm.get_params('r_squared')

# IRASA OSc
ir = (freq, i_sum, "r")
ir_ap = (freq, aperiodic, "darkgreen")
fit_orig = (10**offset / freqs[mask] ** slope)
irasa_osc_orig = np.log10(psd[mask]) - np.log10(fit_orig)



# % Plot Both


fig, axes = plt.subplots(3, 3, figsize=[20, 15])

ax = axes[0, 0]
ax.set_title("Original 0-95Hz", fontsize=40)

ax.loglog(*real_full, **kwgs)
ax.loglog(*i_fit, lw=5, label=f"IRASA 1/f={slope:.2f}")
ax.loglog(fm.freqs, 10**ap_fit, "brown", lw=5, label=f"fooof 1/f={exponent:.2f}")
ax.legend()


ax = axes[1, 0]
ax.semilogy(*real_full, **kwgs)
ax.semilogy(*i_fit, lw=5, label=f"IRASA 1/f={slope:.2f}")
ax.semilogy(fm.freqs, 10**ap_fit, "brown", lw=5, label=f"fooof 1/f={exponent:.2f}")
ax.legend()

ax = axes[2, 0]
ax.axis("off")

fooof_para_str = "Fooof params:\n\n"
for key, value in fooof_params.items():
    fooof_para_str += key + ": " + str(value) + "\n"
ax.text(0, .5, fooof_para_str, fontsize=17)

IRASA_para_str = "IRASA params:\n\n"
for key, value in irasa_params.items():
    if key == "hset":
        value = [np.round(val, 2) for val in value]
        value = f"{value[0]} - {value[-1]}, num: {len(value)}"
    IRASA_para_str += key + ": " + str(value) + "\n"
ax.text(0, -.15, IRASA_para_str, fontsize=17)


# fooof
ax = axes[0, 1]
fm.plot(plt_log=True, ax=ax)

ax.set_ylabel("")
freq_name = f"{freq_range_f[0]}-{freq_range_f[1]}Hz"
ax.set_title(f"Fooof: {freq_name}", fontsize=40)
handles, labels = ax.get_legend_handles_labels()
labels[2] = f"Aperiodic 1/f={exponent:.2f}"
ax.legend(handles, labels)
ax.grid(False)
ax.set_yticklabels("")

ax = axes[1, 1]
R2_mean_F = r_squ
ax.set_title(f"fooof R^2 mean: {R2_mean_F:.2f}")
fm.plot(plt_log=False, ax=ax)

ax.grid(False)
ax.set_ylabel("")
ax.set_xlabel("")
ax.set_yticklabels("")
ax.legend(handles, labels)

ax = axes[2, 1]
ax.plot(fm.freqs, fooof_osc_orig, c="k", label="orig - 1/f")
ax.plot(fm.freqs, fooof_osc, c="r", alpha=0.5, lw=4, label="Osc fit")

ax.set_title(f"Welch: {win_sec}s")
ax.set_xlabel('Frequency')
ax.set_ylabel('PSD in a.u.')
ax.set_yticklabels("")
ax.legend()

# IRASA
ax = axes[0, 2]
freq_name = f"{freq_range_i[0]}-{freq_range_i[1]}Hz"
ax.set_title(f"IRASA {freq_name}", fontsize=40)
ax.loglog(*real, **kwgs)
ax.loglog(*ir, lw=4, alpha=0.5, label="Osc Component")
ax.loglog(*ir_ap, label="Aperiodic Component")
ax.loglog(*i_fit, label=f"Aperiodic 1/f {slope:.2f}")
ax.set_xlabel("log(Frequency)")
ax.set_yticklabels("")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

ax = axes[1, 2]
ax.set_title(f"IRASA R^2 mean: {R2_mean_I:.2f}")
ax.semilogy(*real, **kwgs)
ax.semilogy(*ir, lw=3, alpha=0.5)
ax.semilogy(*ir_ap)
ax.semilogy(*i_fit)
ax.legend(handles, labels)
ax.set_yticklabels("")

ax = axes[2, 2]
osc_log = osc*freq**slope
# Normalize
osc_log /= osc_log.max()
irasa_osc_orig /= irasa_osc_orig.max()
ax.plot(freqs[mask], irasa_osc_orig, "k", label="orig - 1/f")
ax.plot(freq, osc_log, "r", alpha=0.5, lw=4, label="Osc component")
ax.set_yticklabels("")

win_sec_i = irasa_params["win_sec"]
hset = irasa_params["hset"]
hset_max = hset[-1]
num = hset.size
ax.set_title(f"Welch: {win_sec_i}s, hmax={hset_max}, num={num}")
ax.set_xlabel("Frequency")
ax.legend()

subj = easy[i][0]+1
ch = ch_names[easy[i][1]]
cond = conds[easy[i][2]]
title = f"Subj {subj} {ch} {cond} easy"
plt.suptitle(title, fontsize=25)
plt.tight_layout()
plt.savefig(f"../plots/real/{freq_name}IRASA_hset_max={irasa_params['hset'][-1]}_{subj}_{ch}_comp.pdf")
plt.show()






# =============================================================================
#       - why osc + fractal sometimes < original?
#       -> for phase phase coupling the oscillations can cancel out
# =============================================================================




# % Vorläufiges Fazit

# =============================================================================
# Fooof sehr easy zu benutzen und zu kontrollieren, transparent und schnell
# und braucht wenig tuning.
# Am Besten 1/f in Oszillationsfreien Bereichen messen (40-60Hz).
# Bei uns: geht nicht, wegen noise floor. Fooof im niedrig frequenten Bereich
# extrem unzuverlässig.
#
# Wir müssen IRASA benutzen. Funktioniert im Großen und
# Ganzen, ABER: kompliziert, schwer zu verstehen, langsamer, Probleme mit bad
# segments. Liefert bei simulierten Daten bessere Ergebnisse als bei Echten.
#
# Bei weiten Peaks muss hset_max hochjustiert werden. Das aber führt zu
# Artefakten wenn der Frequenz Bereich sehr niedrig beginnt. Vermutlich
# wegen highpass Filter. 
# Außerdem darf die freq range nicht zu groß sein, weil dann der noise floor
# rein rutscht.
# Bei kurzem Frequenzbereich gibt es bessere Ergebnisse
# mit kleinem hset_max wegen noise floor.
# Kleines Win_sec (0.5 bis 1s) ermöglicht etwas besseres 1/f fitting aber dafür
# etwas schlechtere Extraktion von Oszillationen.
#
# Gibt es ein Set an Parametern für IRASA dass bei sehr unterschiedlichen
# Spektren funktioniert?
# =============================================================================
