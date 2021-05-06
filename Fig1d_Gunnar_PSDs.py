"""Gunnar PSD Simplified."""
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np

data_path = "../data/Fig1/"
dummy_name = "Paper-Dummy_2016-11-23_F8-F4.npy"
sub_name = "DBS_2018-03-02_STN_L24_rest.npy"

sub = np.load(data_path + sub_name)
dummy = np.load(data_path + dummy_name)

s_rate = 10000.
stn_ch = "STN_L24" # = "lower left" = "ll"
dummy_ch = "F8-F4"

# Calc PSD
f, psd_sub = sig.welch(sub, fs=s_rate, nperseg=s_rate)
f, psd_dummy = sig.welch(dummy, fs=s_rate, nperseg=s_rate)

# %%

plt.loglog(f, psd_sub, label="Sub2 Ch. " + stn_ch)
plt.loglog(f, psd_dummy, label="Dummy Ch. " + dummy_ch)
plt.legend()
plt.ylim([1e-5, 100])
plt.show()
