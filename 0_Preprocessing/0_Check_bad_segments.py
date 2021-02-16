"""
This scripts plots fif data and allows the annotation of bad segments.
"""
import os
import mne
import matplotlib.pyplot as plt

# =============================================================================
# Create MNE Info
# =============================================================================
ch_names = ['SMA', 'leftM1', 'rightM1',
            'STN_R01', 'STN_R12', 'STN_R23',
            'STN_L01', 'STN_L12', 'STN_L23',
            'EMG_R', 'EMG_L',
            'HEOG', 'VEOG',
            'event']
sfreq = 2400
ch_types = ["mag", "mag", "mag",
            "seeg", "seeg", "seeg", "seeg", "seeg", "seeg",
            "emg", "emg",
            "eog", "eog",
            "stim"]

info = mne.create_info(ch_names, sfreq, ch_types, verbose=True)
info["highpass"] = 1
info["lowpass"] = 600

plot_options = {
            "duration": 10,  # 10 on notebook screen
            "n_channels": 15,
            "scalings": dict(mag=400, seeg=12, eog=100, emg=500, stim=1),
            "color": dict(mag='darkblue', seeg='black',
                          eog='grey', emg='brown', stim='k'),
            "group_by": "original"
            }
# %%
# =============================================================================
# LOAD Files
# =============================================================================
cond = "on"
# =============================================================================
# =============================================================================
for subj in range(0, 1):

    path = '../../data/1_raw_clean_fif/rest/subj'
    path_subj = path + f'{subj+1}/{cond}/'  # subjects start at 1
    fnames = os.listdir(path_subj)  # load first file only
    # delete the wird ".DS_Store" files
    real_fnames = [fname for fname in fnames if fname[0] != "."]
    print(f"Subject {subj+1} {cond} has {len(real_fnames)} file(s).")

    raws = []
    for fname in real_fnames:
        raws.append(mne.io.read_raw_fif(path_subj + fname, info))

    for i, raw in enumerate(raws):
        fig = raw.plot(title=real_fnames[i], **plot_options)
        print(raw.info["bads"])

# %%
# =============================================================================
# # make sure to mark the bad segments before continueing the script
1/0
# =============================================================================


# =============================================================================
# # Save as numpy and fif
# =============================================================================
path_fif = f"../../data/1_raw_clean_fif/rest/subj{subj+1}/{cond}/"

if not os.path.exists(path_fif):
    os.makedirs(path_fif)

for raw, fname in zip(raws, real_fnames):
    raw.save(path_fif + fname[:-4] + ".fif")
print(f"Files saved at\n {path_fif}")

for fig in raws:
    plt.close()

# =============================================================================
# ### Optional:
# ### ... test if annotations are correct:
#
# test = mne.io.read_raw_fif(path_fif + fname[:-4] + "_raw.fif", info)
# _ = test.plot(title=fname[:-4], **plot_options)
# =============================================================================
