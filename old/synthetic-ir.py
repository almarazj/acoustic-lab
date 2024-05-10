import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import median_filter
from utils import *


# Load first RIR
irs_path, filenames = load_rirs_path()
t, rir, fs, duration = load_ir(irs_path, filenames[0])

# Generate synthetic RIR
norm_synt_ir = generate_synthetic_rir(rir, fs)
cutoff_freq = 1000

det, stoc = split_ir(rir, fs, cutoff_freq, 6)
_, synt_stoc = split_ir(norm_synt_ir, fs, cutoff_freq, 6)
filtered_synt_stoc = filter_synthetic_rir(norm_synt_ir, fs, cutoff_freq, 6)
# Filter synthetic IR

# rir_spec = stft(rir, 256, fs)
# synth_spec = stft(norm_synt_ir, 256, fs)

# freq_response = rir_spec[:,0]
# spec_diff = synth_spec[:,0] - freq_response

# filt_synth_spec = synth_spec
# for i in range(np.shape(synth_spec)[1]):
#     filt_synth_spec[:,i] = synth_spec[:,i] - spec_diff

# filt_norm_synt_ir = istft(filt_synth_spec, norm_synt_ir, 256, fs)

# exportar archivos

#_, audio_signal = sig.istft(filt_synt_rir_spec_dB, nperseg=win_size)

sf.write('audio-files/det.wav', det, fs)
sf.write('audio-files/stoc.wav', stoc, fs)
sf.write('audio-files/reconstructed-rir.wav', stoc+det, fs)
sf.write('audio-files/mixed-rir.wav', det+filtered_synt_stoc, fs)
sf.write('audio-files/trimmed-rir.wav', rir, fs)
sf.write('audio-files/synt_stoc.wav', synt_stoc, fs)
sf.write('audio-files/filtered_synt_stoc.wav', filtered_synt_stoc, fs)


