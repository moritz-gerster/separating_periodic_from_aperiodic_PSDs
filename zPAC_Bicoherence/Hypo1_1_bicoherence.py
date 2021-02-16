
# -*- coding: utf-8 -*-
"""
GOAL: MEASURE PAC=BICOHERENCE ZWISCHEN STN UND CORTEX!!!
"""
import os
import numpy as np
import scipy.io
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import csd_helper
from bicoherence import bispectrum

# %%

####################################
# apply some settings for plotting #
####################################
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 9
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()
blind_ax = dict(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelright=False, labeltop=False,
                labelbottom=False)



# %%
# =============================================================================
# LOAD SUBJECTS
# =============================================================================
n_sub = 14
n_ch = 9

on_data = []
on_t = []
on_labels = []
cond = "on"

path = '../../data/raw/rest/subj'

for subj in range(n_sub):
    path_subj = path + f'{subj+1}/{cond}/'  # subjects start at 1
    fname = os.listdir(path_subj)[0]  # load first file only
    if fname == ".DS_Store":
        fname = os.listdir(path_subj)[1]
    data_subj = scipy.io.loadmat(path_subj + fname)['data'][0,0][1][0,0][:n_ch]
    labels_subj = [i[0][0] for i in scipy.io.loadmat(path_subj + fname)['data'][0,0][0][:n_ch]]
    time_subj = scipy.io.loadmat(path_subj + fname)['data'][0,0][2][0,0][0][:n_ch]

    on_data.append(data_subj)
    on_labels.append(labels_subj)
    on_t.append(time_subj)

    if np.allclose(np.diff(on_t[subj]), np.diff(on_t[subj])[0]):
        on_d = np.diff(on_t[subj])[0]
        on_s_rate = 1./on_d
    else:
        raise ValueError('Signal must be evenly sampled')

off_data = []
off_t = []
off_labels = []
cond = "off"

for subj in range(n_sub):
    path_subj = path + f'{subj+1}/{cond}/'  # subjects start at 1
    fname = os.listdir(path_subj)[0]  # load first file only
    if fname == ".DS_Store":
        fname = os.listdir(path_subj)[1]
    data_subj = scipy.io.loadmat(path_subj + fname)['data'][0,0][1][0,0][:n_ch]
    labels_subj = [i[0][0] for i in scipy.io.loadmat(path_subj + fname)['data'][0,0][0][:n_ch]]
    time_subj = scipy.io.loadmat(path_subj + fname)['data'][0,0][2][0,0][0][:n_ch]

    off_data.append(data_subj)
    off_labels.append(labels_subj)
    off_t.append(time_subj)

    if np.allclose(np.diff(off_t[subj]), np.diff(off_t[subj])[0]):
        off_d = np.diff(off_t[subj])[0]
        off_s_rate = 1./off_d
    else:
        raise ValueError('Signal must be evenly sampled')

if not on_labels == off_labels:
    raise ValueError('channel labels must be equal during on and off')



# %%
# =============================================================================
# Welch Parameters
# =============================================================================
n_per_seg = int(0.5 * on_s_rate)
n_overlap = n_per_seg // 2  # previously 0 but makes no difference
average = lambda x: csd_helper.trimmed_mean_cov(x, (0,0.25))


####################################
# calculate spectrum and coherence #
####################################

on_f = []
on_csd = []
on_coherence = []

off_f = []
off_csd = []
off_coherence = []

# include only first 180s
end = int(on_s_rate * 180)

for subj in range(n_sub):

    on_f_sub, on_csd_sub = csd_helper.calc_csd(on_data[subj][:, :end], fs=on_s_rate,
                                    nperseg=n_per_seg, noverlap=n_overlap,
                                    average=average)  # Problem: "median" > 1
    on_coherence_sub = csd_helper.calc_coherence(on_csd_sub)

    on_f.append(on_f_sub)
    on_csd.append(on_csd_sub)
    on_coherence.append(on_coherence_sub)

    off_f_sub, off_csd_sub = csd_helper.calc_csd(off_data[subj][:, :end], fs=off_s_rate,
                                      nperseg=n_per_seg, noverlap=n_overlap,
                                      average=average)
    off_coherence_sub = csd_helper.calc_coherence(off_csd_sub)

    off_f.append(off_f_sub)
    off_csd.append(off_csd_sub)
    off_coherence.append(off_coherence_sub)




# %%
plt.plot(on_f[0], on_csd[0][1, 1])
plt.xlim([0, 30])
#plt.xscale("log")
plt.yscale("log")

X = np.fft.rfft(on_data[subj][1])
Y = np.fft.rfft(on_data[subj][7])
Z = np.fft.rfft(on_data[subj][7])

X = on_csd[subj][1, 1]
Y = on_csd[subj][7, 7]
Z = on_csd[subj][7, 7]

B, N = bispectrum(X, Y, Z, f_axis=0, t_axis=-1, return_norm=True,
               show_progress=True)







# %%
####################
# plot the results #
####################

subj = 0
#n_ch = on_data.shape[0] - 1 # the last channel is the event label
n_ch = 9 # plot only the first 9 channels
plot_f = 40 # frequency until which the signal is plotted

# plot the linear spectral density
psd_fig = plt.figure(figsize=(3, 11.7))
psd_gs = mpl.gridspec.GridSpec(n_ch + 1,2, width_ratios=(0.1,1))
psd_axes = np.zeros(n_ch, dtype=np.object)
text_axes = np.zeros(n_ch, dtype=np.object)
psd_lines = np.zeros((n_ch, 2), dtype=np.object)
for i in range(n_ch):
    if i == 0:
        psd_axes[i] = psd_fig.add_subplot(psd_gs[i,1])
    else:
        psd_axes[i] = psd_fig.add_subplot(psd_gs[i,1],
                sharex=psd_axes[0], sharey=psd_axes[0])
    psd_axes[i].grid()
    psd_lines[i,0], = psd_axes[i].semilogy(on_f[subj][on_f[subj]<=plot_f], np.abs(
        on_csd[subj][i,i,on_f[subj]<=plot_f]), color=color1)
    psd_lines[i,1], = psd_axes[i].semilogy(off_f[subj][off_f[subj]<=plot_f], np.abs(
        off_csd[subj][i,i,off_f[subj]<=plot_f]), color=color2)
    psd_axes[i].set_ylabel('spectrum')
    text_axes[i] = psd_fig.add_subplot(psd_gs[i,0], frameon=False)
    text_axes[i].tick_params(**blind_ax)
    text_axes[i].text(0.5,0.5, s=on_labels[subj][i].replace('_', '\_'),
            ha='center', va='center')

psd_axes[-1].set_xlabel('frequency (Hz)')
psd_axes[-1].set_xlim([0,plot_f])

# add a dummy axis for the legend
psd_legend_ax = psd_fig.add_subplot(psd_gs[-1,:], frame_on=False)
psd_legend_ax.tick_params(**blind_ax)
psd_legend_ax.legend((psd_lines[0,0], psd_lines[0,1]),
        ('ON levodopa', 'OFF levodopa'), loc='center')

psd_fig.tight_layout()
# psd_fig.savefig('subj1_psd.pdf')











































# %%
# =============================================================================
# Calculate coherence and ImCohy in range of 13-30 Hz
# =============================================================================

on_coh_beta = np.zeros([n_sub, n_ch, n_ch])
on_imcohy_beta = np.zeros([n_sub, n_ch, n_ch])
off_coh_beta = np.zeros([n_sub, n_ch, n_ch])
off_imcohy_beta = np.zeros([n_sub, n_ch, n_ch])

for subj in range(n_sub):
    for i in range(n_ch):
        for j in range(n_ch):

            on_coh_beta[subj, i, j] = np.mean(np.abs(
                                      on_coherence[subj][i, j][6:16]))
            on_imcohy_beta[subj, i, j] = np.mean(np.abs(
                                         on_coherence[subj][i, j][6:16].imag))

            off_coh_beta[subj, i, j] = np.mean(np.abs(
                                       off_coherence[subj][i, j][6:16]))
            off_imcohy_beta[subj, i, j] = np.mean(np.abs(
                                        off_coherence[subj][i, j][6:16].imag))

# %%
all_subs = pd.read_pickle("../../data/dataframe/df_fooof.pkl")

# CTX-STN left: all comb. Left M1, SMA, left STN without combs within STN/CTX
combs_ctx_stn_SMA_l = [(0, 6), (0, 7), (0, 8), (1, 6), (1, 7), (1, 8)]
combs_ctx_stn_SMA_r = [(0, 3), (0, 4), (0, 5), (2, 3), (2, 4), (2, 5)]

combs_ctx_stn_l = [(1, 6), (1, 7), (1, 8)]
combs_ctx_stn_r = [(2, 3), (2, 4), (2, 5)]

# CTX-CTX left: all comb. Left M1, SMA within CTX
combs_ctx_ctx_l = [(0, 1)]
combs_ctx_ctx_r = [(0, 2)]

# STN-STN left: all comb. within STN
combs_stn_stn_l = [(6, 7), (6, 8), (7, 8)]
combs_stn_stn_r = [(3, 4), (3, 5), (4, 5)]


for subj in range(n_sub):
    
    # =============================================================================
    # # ctx_stn
    # =============================================================================
    
    # coh
    
    # on
    coh_ctx_stn_on_l = np.mean([on_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_stn_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "coh_ctx_stn"] = coh_ctx_stn_on_l
    
    
    coh_ctx_stn_on_r = np.mean([on_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_stn_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "coh_ctx_stn"] = coh_ctx_stn_on_r
    
    
    coh_ctx_stn_on_SMA_l = np.mean([on_coh_beta[subj][comb[0]][comb[1]]
                                    for comb in combs_ctx_stn_SMA_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "coh_ctx_stn_SMA"] = coh_ctx_stn_on_SMA_l
    
    
    coh_ctx_stn_on_SMA_r = np.mean([on_coh_beta[subj][comb[0]][comb[1]]
                                    for comb in combs_ctx_stn_SMA_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "coh_ctx_stn_SMA"] = coh_ctx_stn_on_SMA_r
    
    
    # off
    coh_ctx_stn_off_l = np.mean([off_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_stn_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "coh_ctx_stn"] = coh_ctx_stn_off_l
    
    
    coh_ctx_stn_off_r = np.mean([off_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_stn_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "coh_ctx_stn"] = coh_ctx_stn_off_r
    
    coh_ctx_stn_off_SMA_l = np.mean([off_coh_beta[subj][comb[0]][comb[1]]
                                    for comb in combs_ctx_stn_SMA_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "coh_ctx_stn_SMA"] = coh_ctx_stn_off_SMA_l
    
    
    coh_ctx_stn_off_SMA_r = np.mean([off_coh_beta[subj][comb[0]][comb[1]]
                                    for comb in combs_ctx_stn_SMA_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "coh_ctx_stn_SMA"] = coh_ctx_stn_off_SMA_r
    
    # imcohy
    
    # on
    
    imcohy_ctx_stn_on_l = np.mean([on_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_stn_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "imcohy_ctx_stn"] = imcohy_ctx_stn_on_l
    
    
    imcohy_ctx_stn_on_r = np.mean([on_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_stn_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "imcohy_ctx_stn"] = imcohy_ctx_stn_on_r
    
    imcohy_ctx_stn_on_SMA_l = np.mean([on_imcohy_beta[subj][comb[0]][comb[1]]
                                    for comb in combs_ctx_stn_SMA_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "imcohy_ctx_stn_SMA"] = imcohy_ctx_stn_on_SMA_l
    
    
    imcohy_ctx_stn_on_SMA_r = np.mean([on_imcohy_beta[subj][comb[0]][comb[1]]
                                    for comb in combs_ctx_stn_SMA_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "imcohy_ctx_stn_SMA"] = imcohy_ctx_stn_on_SMA_r
    
    
    # off
    
    imcohy_ctx_stn_off_l = np.mean([off_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_stn_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "imcohy_ctx_stn"] = imcohy_ctx_stn_off_l
    
    
    imcohy_ctx_stn_off_r = np.mean([off_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_stn_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "imcohy_ctx_stn"] = imcohy_ctx_stn_off_r
    
    
    imcohy_ctx_stn_off_SMA_l = np.mean([off_imcohy_beta[subj][comb[0]][comb[1]]
                                    for comb in combs_ctx_stn_SMA_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "imcohy_ctx_stn_SMA"] = imcohy_ctx_stn_off_SMA_l
    
    
    imcohy_ctx_stn_off_SMA_r = np.mean([off_imcohy_beta[subj][comb[0]][comb[1]]
                                    for comb in combs_ctx_stn_SMA_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "imcohy_ctx_stn_SMA"] = imcohy_ctx_stn_off_SMA_r
    
    # =============================================================================
    # # ctx_ctx
    # =============================================================================
    
    # coh
    
    # on
    coh_ctx_ctx_on_l = np.mean([on_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_ctx_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "coh_ctx_ctx"] = coh_ctx_ctx_on_l
    
    
    coh_ctx_ctx_on_r = np.mean([on_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_ctx_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "coh_ctx_ctx"] = coh_ctx_ctx_on_r
    
    # off
    coh_ctx_ctx_off_l = np.mean([off_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_ctx_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "coh_ctx_ctx"] = coh_ctx_ctx_off_l
    
    
    coh_ctx_ctx_off_r = np.mean([off_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_ctx_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "coh_ctx_ctx"] = coh_ctx_ctx_off_r
    
    
    # imcohy
    
    # on
    
    imcohy_ctx_ctx_on_l = np.mean([on_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_ctx_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "imcohy_ctx_ctx"] = imcohy_ctx_ctx_on_l
    
    
    imcohy_ctx_ctx_on_r = np.mean([on_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_ctx_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "imcohy_ctx_ctx"] = imcohy_ctx_ctx_on_r
    
    
    # off
    
    imcohy_ctx_ctx_off_l = np.mean([off_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_ctx_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "imcohy_ctx_ctx"] = imcohy_ctx_ctx_off_l
    
    
    imcohy_ctx_ctx_off_r = np.mean([off_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_ctx_ctx_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "imcohy_ctx_ctx"] = imcohy_ctx_ctx_off_r
    
    
    
    # =============================================================================
    # # stn_stn
    # =============================================================================
    
    # coh
    
    # on
    coh_stn_stn_on_l = np.mean([on_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_stn_stn_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "coh_stn_stn"] = coh_stn_stn_on_l
    
    
    coh_stn_stn_on_r = np.mean([on_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_stn_stn_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "coh_stn_stn"] = coh_stn_stn_on_r
    
    
    
    # off
    coh_stn_stn_off_l = np.mean([off_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_stn_stn_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "coh_stn_stn"] = coh_stn_stn_off_l
    
    
    coh_stn_stn_off_r = np.mean([off_coh_beta[subj][comb[0]][comb[1]]
                                for comb in combs_stn_stn_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "coh_stn_stn"] = coh_stn_stn_off_r
    
    
    # imcohy
    
    # on
    
    imcohy_stn_stn_on_l = np.mean([on_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_stn_stn_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "imcohy_stn_stn"] = imcohy_stn_stn_on_l
    
    
    imcohy_stn_stn_on_r = np.mean([on_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_stn_stn_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "on")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "imcohy_stn_stn"] = imcohy_stn_stn_on_r
    
    
    
    # off
    
    imcohy_stn_stn_off_l = np.mean([off_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_stn_stn_l])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "left"))
    all_subs.loc[df_filter, "imcohy_stn_stn"] = imcohy_stn_stn_off_l
    
    
    imcohy_stn_stn_off_r = np.mean([off_imcohy_beta[subj][comb[0]][comb[1]]
                                for comb in combs_stn_stn_r])
    df_filter = ((all_subs.subject == subj + 1)
                 & (all_subs.condition == "off")
                 & (all_subs.hemisphere == "right"))
    all_subs.loc[df_filter, "imcohy_stn_stn"] = imcohy_stn_stn_off_r






# %%

all_subs.to_pickle("../../data/dataframe/df_2_v2.pkl")

