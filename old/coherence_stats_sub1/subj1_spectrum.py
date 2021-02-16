import numpy as np
import scipy
import scipy.io
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import csd_helper
#import bicoherence
import pdb

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
#f_on_data = '../data/1_clean_numpy/rest/subj1/on/subj1_on_R7_raw.npy'
#f_off_data = '../data/1_clean_numpy/rest/subj1/off/subj1_off_R7_raw.npy'
f_on_data = 'subj1_on_R7_raw.npy'
f_off_data = 'subj1_off_R7_raw.npy'

ch_names = ['SMA', 'leftM1', 'rightM1',
            'STN_R01', 'STN_R12', 'STN_R23',
            'STN_L01', 'STN_L12', 'STN_L23']

srate = 2400
on_s_rate  = srate
off_s_rate = srate

on_data = np.load(f_on_data)[:9]
#on_data = scipy.io.loadmat(f_on_data)['data'][0,0][1][0,0]
on_labels = ch_names
on_t = np.arange(0, on_data.shape[1]/srate, step=1/srate)

off_data = np.load(f_off_data)[:9]
#on_data = scipy.io.loadmat(f_on_data)['data'][0,0][1][0,0]
off_labels = ch_names
off_t = np.arange(0, off_data.shape[1]/srate, step=1/srate)


N_bootstrap = 500
# calculate a bootstrap for the coherence between on and of
# the coherence is only calculated using the mean across segments
# for the bootstrap, segments are randomly resampled from on and off condition
# with replacement
mag_stat, imag_stat = csd_helper.bootstrap_coherence_diff(
        on_data[:9], off_data[:9], fs=on_s_rate, nperseg=int(0.25 * on_s_rate),
        axis=-1, N_bootstrap=N_bootstrap)

######################
# Calculate p values #
######################
#test only one triangle of the data (symmetric axis and not diagonal)
ix, iy = np.tril_indices(mag_stat[0].shape[0], -1)
mag_p = np.ones_like(mag_stat[0])
imag_p = np.ones_like(mag_stat[0])

# for calculating the p values take the absolute values for a two-tailed test
mag_p[ix, iy] = csd_helper.stepdown_p(
        np.ravel(np.abs(mag_stat[0][ix, iy])),
        np.abs(mag_stat[1][:,ix,iy]).reshape((N_bootstrap, -1), order='C')
        ).reshape((len(ix), -1), order='C')

imag_p[ix, iy] = csd_helper.stepdown_p(
        np.ravel(np.abs(imag_stat[0][ix, iy])),
        np.abs(imag_stat[1][:,ix,iy]).reshape((N_bootstrap, -1), order='C')
        ).reshape((len(ix), -1), order='C')

####################################
# calculate spectrum and coherence #
####################################
on_f, on_csd  = csd_helper.calc_csd(
        on_data[:9], fs=on_s_rate, nperseg=int(0.25*on_s_rate), axis=-1)
on_coherence = csd_helper.calc_coherence(on_csd)

off_f, off_csd = csd_helper.calc_csd(
        off_data[:9], fs=off_s_rate, nperseg=int(0.25*off_s_rate), axis=-1)
off_coherence = csd_helper.calc_coherence(off_csd)

1/0
# =============================================================================
# 
# ################################
# # calculate the 1d bicoherence #
# ################################
# on_f, _, on_spectra = scipy.signal.spectral._spectral_helper(
#         on_data[:9], on_data[:9], fs=on_s_rate, nperseg=int(0.5*off_s_rate),
#         axis=-1)
# # now we have (windowed) spectrum segments with shape
# # channels x frequencies x time -> frequency axis is 1 and time axis is 2
# 
# # we remove the 10% of trials with largest power
# on_spectra_power = (np.abs(on_spectra)**2)[:,1:,:].sum(0).sum(0)
# # 
# on_spectra_mask = (np.argsort(np.argsort(on_spectra_power)) <
#         (0.9*len(on_spectra_power)))
# 
# # making a copy appears to speed up all the calculations
# on_spectra_masked = on_spectra[...,on_spectra_mask].copy()
# 
# import time
# start_time1 = time.time()
# # calculate the bicoherence for single channels
# on_1d_bispectrum, on_1d_bispectrum_norm = bicoherence.bispectrum(
#     on_spectra_masked,
#     on_spectra_masked,
#     on_spectra_masked,
#     f_axis=-2, t_axis=-1) # for axes, see above
# elapsed1 = time.time() - start_time1
# 
# # on_1d_bispectrum is now channels x frequencies x frequencies
# 
# # plot bicoherence in channel 0
# fig = plt.figure()
# ax = fig.add_subplot(111, aspect='equal')
# pc = ax.pcolormesh(on_f, on_f, np.abs(on_1d_bispectrum/on_1d_bispectrum_norm)[0])
# ax.set_xlim([0,300])
# ax.set_ylim([0,300])
# ax.set_xlabel('frequency 1')
# ax.set_ylabel('frequency 2')
# 
# plt.colorbar(pc, ax=ax)
# 
# ###################################################
# # calculate the 2d asymmetric part of bicoherence #
# ###################################################
# idx1, idx2 = np.indices([9, 9])
# 
# # on_spectra[idx] is now channels x channels x frequency x time
# # so frequency is axis 2 and time is axis 3
# 
# start_time2 = time.time()
# # we are now calculating bicoherence_kmm
# on_2d_bispectrum1, on_2d_bispectrum_norm1 = bicoherence.bispectrum(
#         on_spectra_masked[idx1],
#         on_spectra_masked[idx2],
#         on_spectra_masked[idx2],
#         f_axis = -2, t_axis = -1, return_norm = True)
# 
# # we are now calculating bicoherence_mkm
# on_2d_bispectrum2, on_2d_bispectrum_norm2 = bicoherence.bispectrum(
#         on_spectra_masked[idx1],
#         on_spectra_masked[idx2],
#         on_spectra_masked[idx1],
#         f_axis = -2, t_axis = -1, return_norm = True)
# elapsed2 = time.time() - start_time2
# 
# asymmetric_bicoherence = (on_2d_bispectrum1/on_2d_bispectrum_norm1 -
#         on_2d_bispectrum2/on_2d_bispectrum_norm2)
# 
# # plot bicoherence in channel 0 vs channel 1
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, aspect='equal')
# pc2 = ax2.pcolormesh(on_f, on_f, np.abs(asymmetric_bicoherence)[0,1])
# ax2.set_xlim([0,300])
# ax2.set_ylim([0,300])
# ax2.set_xlabel('frequency 1')
# ax2.set_ylabel('frequency 2')
# 
# plt.colorbar(pc2, ax=ax2)
# plt.show()
# 1/0
# =============================================================================


####################
# plot the results #
####################
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
    psd_lines[i,0], = psd_axes[i].semilogy(on_f[on_f<=plot_f], np.abs(
        on_csd[i,i,on_f<=plot_f]), color=color1)
    psd_lines[i,1], = psd_axes[i].semilogy(off_f[off_f<=plot_f], np.abs(
        off_csd[i,i,off_f<=plot_f]), color=color2)
    psd_axes[i].set_ylabel('spectrum')
    text_axes[i] = psd_fig.add_subplot(psd_gs[i,0], frameon=False)
    text_axes[i].tick_params(**blind_ax)
    text_axes[i].text(0.5,0.5, s=on_labels[i].replace('_', '\_'),
            ha='center', va='center')

psd_axes[-1].set_xlabel('frequency (Hz)')
psd_axes[-1].set_xlim([0,plot_f])

# add a dummy axis for the legend
psd_legend_ax = psd_fig.add_subplot(psd_gs[-1,:], frame_on=False)
psd_legend_ax.tick_params(**blind_ax)
psd_legend_ax.legend((psd_lines[0,0], psd_lines[0,1]),
        ('ON levodopa', 'OFF levodopa'), loc='center')

psd_fig.tight_layout()
psd_fig.savefig('subj1_psd.pdf')

######################
# plot the coherence #
######################
csd_fig = plt.figure(figsize=(25, 25))
csd_gs = mpl.gridspec.GridSpec(n_ch + 2, n_ch + 1,
        width_ratios=np.r_[0.1, [1]*n_ch],
        height_ratios = np.r_[0.1, [1]*(n_ch + 1)])
csd_axes = np.zeros([n_ch, n_ch], dtype=np.object)
csd_text_axes = np.zeros([n_ch, 2], dtype=np.object)
csd_lines = np.zeros((n_ch, n_ch, 4), dtype=np.object)
for i in range(n_ch):
    for j in range(n_ch):
        if (i == 0) & (j == 0):
            csd_axes[i,j] = csd_fig.add_subplot(csd_gs[i + 1, j + 1])
        else:
            csd_axes[i,j] = csd_fig.add_subplot(csd_gs[i + 1, j + 1],
                    sharex=csd_axes[0,0], sharey=csd_axes[0,0])
        csd_axes[i,j].grid()
        csd_lines[i,j,0], = csd_axes[i,j].plot(on_f[on_f<=plot_f], np.abs(
            on_coherence[i,j,on_f<=plot_f]), color=color1, ls='-', lw=2)
        csd_lines[i,j,1], = csd_axes[i,j].plot(on_f[on_f<=plot_f], np.abs(
            on_coherence[i,j,on_f<=plot_f].imag), color=color1, ls='-', lw=1)
        csd_lines[i,j,2], = csd_axes[i,j].plot(off_f[off_f<=plot_f], np.abs(
            off_coherence[i,j,off_f<=plot_f]), color=color2, ls='-', lw=2)
        csd_lines[i,j,3], = csd_axes[i,j].plot(off_f[off_f<=plot_f], np.abs(
            off_coherence[i,j,off_f<=plot_f].imag), color=color2, ls='-', lw=1)
        if i == 0:
            csd_text_axes[j,0] = csd_fig.add_subplot(csd_gs[i,j + 1], frameon=False)
            csd_text_axes[j,0].tick_params(**blind_ax)
            csd_text_axes[j,0].text(0.5,0.5, s=on_labels[j].replace('_', '\_'),
                    ha='center', va='center', fontsize=12)
        if i == n_ch - 1:
            csd_axes[i,j].set_xlabel('frequency')
        if j == 0:
            csd_axes[i,j].set_ylabel('spectrum')
            csd_text_axes[i,1] = csd_fig.add_subplot(csd_gs[i + 1,j], frameon=False)
            csd_text_axes[i,1].tick_params(**blind_ax)
            csd_text_axes[i,1].text(0.5,0.5, s=on_labels[i].replace('_', '\_'),
                    ha='center', va='center', fontsize=12)

csd_axes[0,0].set_xlim([0,plot_f])
csd_axes[0,0].set_ylim([0,1])

# add a dummy axis for the legend
csd_legend_ax = csd_fig.add_subplot(csd_gs[-1,:], frame_on=False)
csd_legend_ax.tick_params(**blind_ax)
csd_legend_ax.legend(
        (csd_lines[0,0,0],
            csd_lines[0,0,1],
            csd_lines[0,0,2],
            csd_lines[0,0,3]),
        ('ON levodopa (magnitude coherence)',
            'ON levodopa (absolute imaginary coherence)',
            'OFF levodopa (magnitude coherence)',
            'OFF levodopa (absolute imaginary coherence)'),
        loc='center', ncol=2)

csd_fig.tight_layout()
csd_fig.savefig('subj1_csd.pdf')
