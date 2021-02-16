import numpy as np
import scipy
import scipy.io
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import csd_helper
import bicoherence

import time

####################################
# apply some settings for plotting #
####################################
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 9
cmap = 'plasma'
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()
blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

#################
# read the data #
#################
f_on_data = '../../data/raw/rest/subj1_old/on/R1.mat'
f_off_data = '../../data/raw/rest/subj1_old/off/R1.mat'

on_data = scipy.io.loadmat(f_on_data)['data'][0,0][1][0,0]
on_labels = [i[0][0] for i in scipy.io.loadmat(f_on_data)['data'][0,0][0]]
on_t = scipy.io.loadmat(f_on_data)['data'][0,0][2][0,0][0]

off_data = scipy.io.loadmat(f_off_data)['data'][0,0][1][0,0]
off_labels = [i[0][0] for i in scipy.io.loadmat(f_off_data)['data'][0,0][0]]
off_t = scipy.io.loadmat(f_off_data)['data'][0,0][2][0,0][0]

if np.allclose(np.diff(on_t), np.diff(on_t)[0]):
    on_d = np.diff(on_t)[0]
    on_s_rate = 1./on_d
else:
    raise ValueError('Signal must be evenly sampled')

if np.allclose(np.diff(off_t), np.diff(off_t)[0]):
    off_d = np.diff(off_t)[0]
    off_s_rate = 1./off_d
else:
    raise ValueError('Signal must be evenly sampled')

if not on_labels == off_labels:
    raise ValueError('channel labels must be equal during on and off')

####################################
# calculate spectrum and coherence #
####################################
# =============================================================================
# on_f, on_csd = csd_helper.calc_csd(
#         on_data[:9], fs=on_s_rate, nperseg=int(0.5*on_s_rate), axis=-1,
#         average=lambda x: csd_helper.trimmed_mean_cov(x, (0,0.25)))
# on_coherence = csd_helper.calc_coherence(on_csd)
# off_f, off_csd = csd_helper.calc_csd(
#         off_data[:9], fs=off_s_rate, nperseg=int(0.5*off_s_rate), axis=-1,
#         average=lambda x: csd_helper.trimmed_mean_cov(x, (0,0.25)))
# off_coherence = csd_helper.calc_coherence(off_csd)
# =============================================================================

# %%
end = int(180 * on_s_rate)
################################
# calculate the 1d bicoherence #
################################
# =============================================================================
# scipy.signal.spectral._spectral_helper = signal.welch
# =============================================================================
on_f, _, on_spectra = scipy.signal.spectral._spectral_helper(
        on_data[:9, :end], on_data[:9, :end], fs=on_s_rate, nperseg=int(0.5*on_s_rate),
        axis=-1)

off_f, _, off_spectra = scipy.signal.spectral._spectral_helper(
        off_data[:9, :end], off_data[:9, :end], fs=off_s_rate, nperseg=int(0.5*off_s_rate),
        axis=-1)
# %%
# now we have (windowed) spectrum segments with shape
# channels x frequencies x time -> frequency axis is 1 and time axis is 2

# =============================================================================
# WHY HAS TIME 719 ENTRIES? overlap = nperseg // 2 = 600
# -> times / overlap = 719!
# =============================================================================

# we remove the 10% of trials with largest power
on_spectra_power = (np.abs(on_spectra)**2)[:,1:,:].sum(0).sum(0)
off_spectra_power = (np.abs(off_spectra)**2)[:,1:,:].sum(0).sum(0)

# 
# =============================================================================
# DOPPELTES np.argsort MACHT KEIN SINN
# =============================================================================
on_spectra_mask = (np.argsort(np.argsort(on_spectra_power)) <
        (0.9*len(on_spectra_power)))
off_spectra_mask = (np.argsort(np.argsort(off_spectra_power)) <
        (0.9*len(off_spectra_power)))

start = time.perf_counter()
# calculate the bicoherence for single channels
on_1d_bispectrum, on_1d_bispectrum_norm = bicoherence.bispectrum(
    on_spectra[...,on_spectra_mask],
    on_spectra[...,on_spectra_mask],
    on_spectra[...,on_spectra_mask],
    f_axis=1, t_axis=2) # for axes, see above
end = time.perf_counter()
print(end-start)
off_1d_bispectrum, off_1d_bispectrum_norm = bicoherence.bispectrum(
    off_spectra[...,on_spectra_mask],
    off_spectra[...,on_spectra_mask],
    off_spectra[...,on_spectra_mask],
    f_axis=1, t_axis=2) # for axes, see above

# =============================================================================
# on_spectra[...,on_spectra_mask] == on_spectra[:,:,on_spectra_mask]
# =============================================================================
# =============================================================================
# BICOHRENCE of 9 electrodes within themselves takes 2 minutes -> now 40s
# =============================================================================

# on_1d_bispectrum is now channels x frequencies x frequencies
# %%
fig, axes = plt.subplots(3, 6, figsize=(20,10), sharex=True, sharey=True)
#ax = ax.flatten()
plt.rcParams.update({'font.size': 10})


for ch in range(9):

    ax = axes[ch//3, ch%3]
    pc = ax.pcolormesh(on_f, on_f, np.abs(on_1d_bispectrum/on_1d_bispectrum_norm)[ch])
    ax.set_xlim([0, 300])
    ax.set_ylim([0, 300])
    ax.set_title(on_labels[ch], fontsize=15)
    ax.tick_params(axis="both", labelsize=15)
    if ch//3 == 2:
        ax.set_xlabel('frequency 1', fontsize=15)
    if ch%3 == 0:
        ax.set_ylabel('frequency 2', fontsize=15)

    ax = axes[ch//3, ch%3+3]
    pc = ax.pcolormesh(off_f, off_f, np.abs(off_1d_bispectrum/off_1d_bispectrum_norm)[ch])
    ax.set_xlim([0, 300])
    ax.set_ylim([0, 300])
    ax.set_title(off_labels[ch], fontsize=15)
    ax.tick_params(axis="both", labelsize=15)
    if ch//3 == 2:
        ax.set_xlabel('frequency 1', fontsize=15)
    if ch%3 == 0:
        ax.set_ylabel('frequency 2', fontsize=15)

plt.suptitle("Subj 0", fontsize=30, position=(0.5, 1.02))
axes[0, 1].text(0.4, 1.2, "On", fontsize=25, transform=axes[0, 1].transAxes)
axes[0, 4].text(0.4, 1.2, "Off", fontsize=25, transform=axes[0, 4].transAxes)
plt.tight_layout()
cb = plt.colorbar(pc, ax=axes[:, 5]).ax.tick_params(axis="y", labelsize=15)
#plt.savefig("../../plots/bicoherence_subj0_single.png", bbox_inches="tight")
plt.show()

# %%
for ch in range(9):
    
    # plot bicoherence in channel 0
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pc = ax.pcolormesh(on_f, on_f, np.abs(on_1d_bispectrum/on_1d_bispectrum_norm)[ch])
    ax.set_xlim([0,300])
    ax.set_ylim([0,300])
    ax.set_title("Subj 0 On " + on_labels[ch])
    ax.set_xlabel('frequency 1')
    ax.set_ylabel('frequency 2')
    plt.colorbar(pc, ax=ax)
    plt.show()

# %%
for ch in range(9):
    
    # plot bicoherence in channel 0
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pc = ax.pcolormesh(off_f, off_f, np.abs(off_1d_bispectrum/off_1d_bispectrum_norm)[ch])
    ax.set_xlim([0,300])
    ax.set_ylim([0,300])
    ax.set_title("Subj 0 Off " + off_labels[ch])
    ax.set_xlabel('frequency 1')
    ax.set_ylabel('frequency 2')
    plt.colorbar(pc, ax=ax)
    plt.show()
# %%
###################################################
# calculate the 2d asymmetric part of bicoherence #
###################################################
# %%
# uncomment to calculate between all channels (takes ~1.5 h on my computer)
idx1, idx2 = np.indices([9, 9])

# on_spectra[idx] is now channels x channels x frequency x time
# so frequency is axis 2 and time is axis 3
start = time.perf_counter()
# we are now calculating bicoherence_kmm
on_2d_bispectrum1, on_2d_bispectrum_norm1 = bicoherence.bispectrum(
        on_spectra[idx1][...,on_spectra_mask],
        on_spectra[idx2][...,on_spectra_mask],
        on_spectra[idx2][...,on_spectra_mask],
        f_axis = 2, t_axis = 3, return_norm = True)

# we are now calculating bicoherence_mkm
on_2d_bispectrum2, on_2d_bispectrum_norm2 = bicoherence.bispectrum(
        on_spectra[idx2][...,on_spectra_mask],
        on_spectra[idx1][...,on_spectra_mask],
        on_spectra[idx2][...,on_spectra_mask],
        f_axis = 2, t_axis = 3, return_norm = True)

on_asymmetric_bicoherence = (on_2d_bispectrum1/on_2d_bispectrum_norm1 -
        on_2d_bispectrum2/on_2d_bispectrum_norm2)

end = time.perf_counter()
print(end-start)
# =============================================================================
# BICOHRENCE of 9*9 electrodes takes 90 minutes
# =============================================================================
# %%
# take non repeating combinations within the same hemisphere
combs = [(0, 1) ,(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                    (0, 8), (1, 2), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4),
                    (2, 5), (3, 4), (3, 5), (4, 5), (6, 7), (6, 8), (7, 8)]

for comb in combs:

# plot bicoherence in channel 0 vs channel 1
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    pc2 = ax2.pcolormesh(on_f, on_f, np.abs(on_asymmetric_bicoherence)[comb[0],comb[1]],
                         vmin=0, vmax=1)
    ax2.set_xlim([0,300])
    ax2.set_ylim([0,300])
    ax2.set_title("Subj 0 On " + on_labels[comb[0]] + " - "+ on_labels[comb[1]] + f" {comb[0]} {comb[1]}")
    ax2.set_xlabel('frequency 1')
    ax2.set_ylabel('frequency 2')
    plt.colorbar(pc2, ax=ax2)
    plt.show()

# %%
combs = [(4, 5), (6, 7), (0,1), (3, 5)]
names = ["No PAC", "Pattern", "Diffuse", "real PAC?"]
for comb, name in zip(combs, names):

# plot bicoherence in channel 0 vs channel 1
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    pc2 = ax2.pcolormesh(on_f, on_f, np.abs(on_asymmetric_bicoherence)[comb[0],comb[1]],
                         vmin=0, vmax=1)
    ax2.set_xlim([0,300])
    ax2.set_ylim([0,300])
    ax2.set_title("Subj 0 On " + on_labels[comb[0]] + " - "+ on_labels[comb[1]] + " " + name)
    ax2.set_xlabel('frequency 1')
    ax2.set_ylabel('frequency 2')
    plt.colorbar(pc2, ax=ax2)
    plt.show()
#%%


        
# %%
###################################################
# calculate the 2d asymmetric part of bicoherence #
###################################################
# %%
# uncomment to calculate between all channels (takes ~1.5 h on my computer)
idx1, idx2 = np.indices([9, 9])

# on_spectra[idx] is now channels x channels x frequency x time
# so frequency is axis 2 and time is axis 3
start = time.perf_counter()
# we are now calculating bicoherence_kmm
off_2d_bispectrum1, off_2d_bispectrum_norm1 = bicoherence.bispectrum(
        off_spectra[idx1][...,off_spectra_mask],
        off_spectra[idx2][...,off_spectra_mask],
        off_spectra[idx2][...,off_spectra_mask],
        f_axis = 2, t_axis = 3, return_norm = True,
        show_progress = True)

# we are now calculating bicoherence_mkm
off_2d_bispectrum2, off_2d_bispectrum_norm2 = bicoherence.bispectrum(
        off_spectra[idx2][...,off_spectra_mask],
        off_spectra[idx1][...,off_spectra_mask],
        off_spectra[idx2][...,off_spectra_mask],
        f_axis = 2, t_axis = 3, return_norm = True,
        show_progress = True)

off_asymmetric_bicoherence = (off_2d_bispectrum1/off_2d_bispectrum_norm1 -
        off_2d_bispectrum2/off_2d_bispectrum_norm2)

end = time.perf_counter()
print(end-start)
# =============================================================================
# BICOHRENCE of 9*9 electrodes takes 90 minutes
# =============================================================================
# %%
# take non repeating combinations within the same hemisphere
combs = [(0, 1) ,(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                    (0, 8), (1, 2), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4),
                    (2, 5), (3, 4), (3, 5), (4, 5), (6, 7), (6, 8), (7, 8)]

for comb in combs:

# plot bicoherence in channel 0 vs channel 1
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    pc2 = ax2.pcolormesh(off_f, off_f, np.abs(off_asymmetric_bicoherence)[comb[0],comb[1]],
                         vmin=0, vmax=1)
    ax2.set_xlim([0,300])
    ax2.set_ylim([0,300])
    ax2.set_title("Subj 0 Off " + off_labels[comb[0]] + " - "+ off_labels[comb[1]] + f" {i} {comb[1]}")
    ax2.set_xlabel('frequency 1')
    ax2.set_ylabel('frequency 2')
    plt.colorbar(pc2, ax=ax2)
    plt.show()
# %%
1/0

# =============================================================================
# 
# ####################
# # plot the results #
# ####################
# #n_ch = on_data.shape[0] - 1 # the last channel is the event label
# n_ch = 9 # plot only the first 9 channels
# plot_f = 40 # frequency until which the signal is plotted
# 
# # plot the linear spectral density
# psd_fig = plt.figure(figsize=(3, 11.7))
# psd_gs = mpl.gridspec.GridSpec(n_ch + 1,2, width_ratios=(0.1,1))
# psd_axes = np.zeros(n_ch, dtype=np.object)
# text_axes = np.zeros(n_ch, dtype=np.object)
# psd_lines = np.zeros((n_ch, 2), dtype=np.object)
# for i in range(n_ch):
#     if i == 0:
#         psd_axes[i] = psd_fig.add_subplot(psd_gs[i,1])
#     else:
#         psd_axes[i] = psd_fig.add_subplot(psd_gs[i,1],
#                 sharex=psd_axes[0], sharey=psd_axes[0])
#     psd_axes[i].grid()
#     psd_lines[i,0], = psd_axes[i].semilogy(on_f[on_f<=plot_f], np.abs(
#         on_csd[i,i,on_f<=plot_f]), color=color1)
#     psd_lines[i,1], = psd_axes[i].semilogy(off_f[off_f<=plot_f], np.abs(
#         off_csd[i,i,off_f<=plot_f]), color=color2)
#     psd_axes[i].set_ylabel('spectrum')
#     text_axes[i] = psd_fig.add_subplot(psd_gs[i,0], frameon=False)
#     text_axes[i].tick_params(**blind_ax)
#     text_axes[i].text(0.5,0.5, s=on_labels[i].replace('_', '\_'),
#             ha='center', va='center')
# 
# psd_axes[-1].set_xlabel('frequency (Hz)')
# psd_axes[-1].set_xlim([0,plot_f])
# 
# # add a dummy axis for the legend
# psd_legend_ax = psd_fig.add_subplot(psd_gs[-1,:], frame_on=False)
# psd_legend_ax.tick_params(**blind_ax)
# psd_legend_ax.legend((psd_lines[0,0], psd_lines[0,1]),
#         ('ON levodopa', 'OFF levodopa'), loc='center')
# 
# psd_fig.tight_layout()
# #psd_fig.savefig('subj1_psd.pdf')
# 
# ######################
# # plot the coherence #
# ######################
# csd_fig = plt.figure(figsize=(25, 25))
# csd_gs = mpl.gridspec.GridSpec(n_ch + 2, n_ch + 1,
#         width_ratios=np.r_[0.1, [1]*n_ch],
#         height_ratios = np.r_[0.1, [1]*(n_ch + 1)])
# csd_axes = np.zeros([n_ch, n_ch], dtype=np.object)
# csd_text_axes = np.zeros([n_ch, 2], dtype=np.object)
# csd_lines = np.zeros((n_ch, n_ch, 4), dtype=np.object)
# for i in range(n_ch):
#     for j in range(n_ch):
#         if (i == 0) & (j == 0):
#             csd_axes[i,j] = csd_fig.add_subplot(csd_gs[i + 1, j + 1])
#         else:
#             csd_axes[i,j] = csd_fig.add_subplot(csd_gs[i + 1, j + 1],
#                     sharex=csd_axes[0,0], sharey=csd_axes[0,0])
#         csd_axes[i,j].grid()
#         csd_lines[i,j,0], = csd_axes[i,j].plot(on_f[on_f<=plot_f], np.abs(
#             on_coherence[i,j,on_f<=plot_f]), color=color1, ls='-', lw=2)
#         csd_lines[i,j,1], = csd_axes[i,j].plot(on_f[on_f<=plot_f], np.abs(
#             on_coherence[i,j,on_f<=plot_f].imag), color=color1, ls='-', lw=1)
#         csd_lines[i,j,2], = csd_axes[i,j].plot(off_f[off_f<=plot_f], np.abs(
#             off_coherence[i,j,off_f<=plot_f]), color=color2, ls='-', lw=2)
#         csd_lines[i,j,3], = csd_axes[i,j].plot(off_f[off_f<=plot_f], np.abs(
#             off_coherence[i,j,off_f<=plot_f].imag), color=color2, ls='-', lw=1)
#         if i == 0:
#             csd_text_axes[j,0] = csd_fig.add_subplot(csd_gs[i,j + 1], frameon=False)
#             csd_text_axes[j,0].tick_params(**blind_ax)
#             csd_text_axes[j,0].text(0.5,0.5, s=on_labels[j].replace('_', '\_'),
#                     ha='center', va='center', fontsize=12)
#         if i == n_ch - 1:
#             csd_axes[i,j].set_xlabel('frequency')
#         if j == 0:
#             csd_axes[i,j].set_ylabel('spectrum')
#             csd_text_axes[i,1] = csd_fig.add_subplot(csd_gs[i + 1,j], frameon=False)
#             csd_text_axes[i,1].tick_params(**blind_ax)
#             csd_text_axes[i,1].text(0.5,0.5, s=on_labels[i].replace('_', '\_'),
#                     ha='center', va='center', fontsize=12)
# 
# csd_axes[0,0].set_xlim([0,plot_f])
# csd_axes[0,0].set_ylim([0,1])
# 
# # add a dummy axis for the legend
# csd_legend_ax = csd_fig.add_subplot(csd_gs[-1,:], frame_on=False)
# csd_legend_ax.tick_params(**blind_ax)
# csd_legend_ax.legend(
#         (csd_lines[0,0,0],
#             csd_lines[0,0,1],
#             csd_lines[0,0,2],
#             csd_lines[0,0,3]),
#         ('ON levodopa (magnitude coherence)',
#             'ON levodopa (absolute imaginary coherence)',
#             'OFF levodopa (magnitude coherence)',
#             'OFF levodopa (absolute imaginary coherence)'),
#         loc='center', ncol=2)
# 
# csd_fig.tight_layout()
# #csd_fig.savefig('subj1_csd.pdf')
# 
# =============================================================================
