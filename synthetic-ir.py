import os
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import interpolation
from scipy.signal import hilbert, find_peaks, istft, medfilt
from utils import *

# load rirs path
path = os.path.abspath(os.getcwd())
irs_path= os.path.join(path,'audio-files')
filenames = [f for f in os.listdir(irs_path) if f.endswith('.wav')]
filenames.sort()

# Load first RIR
t, rir, fs, duration = load_ir(irs_path, filenames[0])

# measured IR envelope

peaks, _ = find_peaks(np.abs(rir))

env_peaks = np.zeros(len(rir))
env_peaks[peaks] = np.abs(rir)[peaks]

z = len(rir) / len(rir[peaks])

ir_env = interpolation.zoom(np.abs(rir)[peaks], z)
ir_env[0] = rir[0]

#------------- Synthetic IR generator -------------#

# M = 0.0005         # background noise amplitude
# A = 0.055           # impulse response amplitude
# rt = 0.5           # reverberation time

# # decay rate
# b = 3 / (rt * np.log(np.e))    

# gaussian noise
noise = np.random.normal(0, 1, int(fs*duration))


# time vector
t = np.linspace(0, int(duration), int(fs*duration))

# synthetic IR
syntetic_ir = (noise * ir_env) 
n = np.max(syntetic_ir) / np.max(rir)
normalized_synthetic_ir = syntetic_ir /n

# Ploteos
plt.plot(t, 0.5+rir, label='rir real')
#plt.plot(t, 1+ir_env, label='envolvente real')
plt.plot(t, -0.5+normalized_synthetic_ir, label='rir sintetica')
plt.legend()
plt.show()

# espectrograma

f, t, Sxx = sig.spectrogram(rir, fs)
Sxx = 20*np.log10(Sxx/2e-5)

f, t, synt_Sxx = sig.spectrogram(normalized_synthetic_ir, fs)
synt_Sxx = 20*np.log10(synt_Sxx/2e-5)

frequency_response = medfilt(Sxx[:,0], 25)

spec_diff = medfilt(synt_Sxx[:,0], 25) - frequency_response

synt_Sxx_new = synt_Sxx
for i in range(np.shape(synt_Sxx)[1]):
    synt_Sxx_new[:, i] = synt_Sxx[:, i] - spec_diff

fig, (ax1, ax2) = plt.subplots(1, 2)
norm = mpl.colors.Normalize(vmin=-100, vmax=0)
mesh1 = ax1.pcolormesh(t, f, Sxx, shading='gouraud', norm=norm)
#ax1.set_yscale('symlog')
ax1.set_ylim(20,20000)
mesh2 = ax2.pcolormesh(t, f, synt_Sxx, shading='gouraud', norm=norm)
#ax2.set_yscale('symlog')
ax2.set_ylim(20,20000)
#ax3.pcolormesh(t, f, synt_Sxx_new, shading='gouraud')
#ax2.set_yscale('symlog')
#ax3.set_ylim(20,20000)
fig.colorbar(mesh1, norm=norm)
fig.colorbar(mesh2, norm=norm)
#fig.colorbar(ax3.pcolormesh(t, f, synt_Sxx_new, shading='gouraud'))

plt.show()


# exportar archivos
_, audio_signal = istft(synt_Sxx_new)

sf.write('RIRs/synthetic_ir.wav', audio_signal, fs)