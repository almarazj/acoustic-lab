import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import istft, medfilt
from utils import *


# Load first RIR
irs_path, filenames = load_rirs_path()
t, rir, fs, duration = load_ir(irs_path, filenames[0])

# Generate synthetic RIR
norm_synt_ir = generate_synthetic_rir(rir)

# Ploteos
# plt.plot(t, 0.5+rir, label='rir real')
# plt.plot(t, -0.5+norm_synt_ir, label='rir sintetica')
# plt.legend()
# plt.show()

# espectrograma
win_size = 256
f, t, rir_spec = sig.stft(rir,
                            fs=fs, 
                            nperseg=win_size, 
                            padded=True)

rir_spec_dB = np.abs(rir_spec)

_, _, synt_rir_spec = sig.stft(norm_synt_ir,
                                fs=fs,
                                nperseg=win_size,
                                padded=True)

synt_rir_spec_dB = np.abs(synt_rir_spec)

frequency_response = medfilt(rir_spec_dB[:,0], 25)

spec_diff = medfilt(synt_rir_spec_dB[:,0], 25) - frequency_response

filt_synt_rir_spec_dB = synt_rir_spec_dB
for i in range(np.shape(synt_rir_spec_dB)[1]):
    filt_synt_rir_spec_dB[:, i] = synt_rir_spec_dB[:, i] - spec_diff

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#norm = mpl.colors.Normalize(vmin=-1, vmax=1)
mesh1 = ax1.pcolormesh(t, f, rir_spec_dB, shading='gouraud')
ax1.set_ylim(20,20000)
mesh2 = ax2.pcolormesh(t, f, synt_rir_spec_dB, shading='gouraud')
ax2.set_ylim(20,20000)
mesh3 = ax3.pcolormesh(t, f, filt_synt_rir_spec_dB, shading='gouraud')
ax3.set_ylim(20,20000)
fig.colorbar(mesh1)
fig.colorbar(mesh2)
fig.colorbar(mesh3)


plt.show()


# exportar archivos
_, audio_signal = istft(filt_synt_rir_spec_dB, nperseg=win_size)

sf.write('audio-files/synthetic_ir.wav', audio_signal, fs)