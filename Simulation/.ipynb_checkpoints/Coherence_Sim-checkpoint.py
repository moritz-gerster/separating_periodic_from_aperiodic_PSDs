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
# simulate sine waves #
#################

srate = 2400
on_s_rate  = srate
off_s_rate = srate

# %%
time = np.arange(181 * srate)

freq = 10 # Hz


signal1 = np.sin(2 * np.pi * freq * time / srate)
signal2 = signal1

plt.plot(signal1)
plt.xticks(time[::600], time[::600] / srate)
plt.xlim([0, 2400])
plt.xlabel("Time in Seconds")
plt.show()

on_signal = np.array([signal1, signal2])

off_signal = on_signal


# %%
####################################
# calculate spectrum and coherence #
####################################
on_f, on_csd  = csd_helper.calc_csd(
        on_signal, fs=on_s_rate, nperseg=int(0.5*on_s_rate), axis=-1)

plt.plot(on_f, np.abs(on_csd[0, 1]))
plt.xlim([0, 20])
plt.title("Power Spectrum")
plt.xlabel("Frequency [Hz]")
plt.show()

# %%

on_coherence = csd_helper.calc_coherence(on_csd)

plt.plot(on_f, np.abs(on_coherence[0, 1]), label="Mag. Coh.")
plt.plot(on_f, np.abs(on_coherence[0, 1].imag), label="ImCohy")
#plt.xlim([0, 20])
plt.legend()
plt.title("Coherence")
plt.show()

# %%

off_f, off_csd = csd_helper.calc_csd(
        off_signal, fs=off_s_rate, nperseg=int(0.5*off_s_rate), axis=-1)
off_coherence = csd_helper.calc_coherence(off_csd)


# %%

N_bootstrap = 500
# calculate a bootstrap for the coherence between on and of
# the coherence is only calculated using the mean across segments
# for the bootstrap, segments are randomly resampled from on and off condition
# with replacement
f_stat, mag_stat, imag_stat = csd_helper.bootstrap_coherence_diff(
        on_signal, off_signal, fs=on_s_rate, nperseg=int(0.5 * on_s_rate),
        axis=-1, N_bootstrap=N_bootstrap, frange = (1,45))

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
# make the array symmetric again
mag_p = mag_p + np.transpose(mag_p, (1,0,2)) - 1

imag_p[ix, iy] = csd_helper.stepdown_p(
        np.ravel(np.abs(imag_stat[0][ix, iy])),
        np.abs(imag_stat[1][:,ix,iy]).reshape((N_bootstrap, -1), order='C')
        ).reshape((len(ix), -1), order='C')
# make the array symmetric again
imag_p = imag_p + np.transpose(imag_p, (1,0,2)) - 1

#
freq_idx = 9


# %%

fig, ax = plt.subplots(ncols=2, nrows=1)
cb0 = ax[0].matshow(mag_p[...,freq_idx], vmin=0, vmax=1)
cb1 = ax[1].matshow(imag_p[...,freq_idx], vmin=0, vmax=1)
ax[0].set_title('p Values for difference of Magnitude Coherence')
ax[1].set_title('p Values for difference of imaginary Coherence')
fig.colorbar(cb0, ax=ax[0])
fig.colorbar(cb1, ax=ax[1])
#ax[0].set_yticklabels(["SMA"])
fig.suptitle('f = {} Hz'.format(f_stat[freq_idx]))
#fig.savefig('p_values_20Hz.png')

# %%

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(on_f, np.abs(on_coherence)[0,1], 'r-', label='ON')
ax.plot(off_f, np.abs(off_coherence)[0,1], 'b-', label='OFF')
ax.scatter(f_stat[mag_p[0,1]<0.05], 0.9*np.ones(np.sum(mag_p[0,1]<0.05)),
    edgecolors='k', facecolors='k')
ax.set_ylim([0, 1])
ax.set_xlim([0,45])
#ax.set_title(r'magnitude coherence channel {0} vs. {1}'.format(
 #   ch_names[0], ch_names[5]))
ax.set_ylabel('frequency (Hz)')
ax.legend()
#fig.savefig('mag_coherence_0_5.png')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(on_f, np.abs(on_coherence.imag)[0,1], 'r-', label='ON')
ax.plot(off_f, np.abs(off_coherence.imag)[0,1], 'b-', label='OFF')
ax.scatter(f_stat[imag_p[0,1]<0.05], 0.9*np.ones(np.sum(imag_p[0,1]<0.05)),
    edgecolors='k', facecolors='k')
ax.set_ylim([0, 1])
ax.set_xlim([0,45])
#ax.set_title(r'absolute imaginary coherence channel {0} vs. {1}'.format(
 #   ch_names[0], ch_names[5]))
ax.set_ylabel('frequency (Hz)')
ax.colorbar()
ax.legend()
#fig.savefig('imag_coherence_0_5.png')


fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].hist(mag_p[...,freq_idx])
ax[1].hist(imag_p[...,freq_idx])
fig.suptitle("Histogram p-values")
ax[0].set_xlabel("Mag. Coh. p-values")
ax[1].set_xlabel("Imag. Coh. p-values")
ax[0].set_ylabel("Count")
#fig.savefig('histogram.png')



TO DO:
    
    - add noise
    - shift phase
    - modulate power differently

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
