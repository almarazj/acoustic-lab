import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import soundfile as sf
import scipy.signal as sig
from scipy.signal import hilbert, find_peaks
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, freqz
from scipy.ndimage import interpolation


# load rirs path
path = os.path.abspath(os.getcwd())
IRs= os.path.join(path,'RIRs')
filenames = [f for f in os.listdir(IRs) if f.endswith('.wav')]
filenames.sort()

# Load first RIR
raw_rir, fs = sf.read(os.path.join(IRs, filenames[0]))

raw_duration = len(raw_rir) / fs

raw_t = np.linspace(0, int(raw_duration), int(fs*raw_duration))

t = raw_t[0 : int(fs*raw_duration)-np.argmax(np.abs(raw_rir))-1]
rir = raw_rir[np.argmax(np.abs(raw_rir)):-1]

duration = len(rir) / fs

# measured IR envelope

peaks, _ = find_peaks(rir, distance=2400)

env_peaks = np.zeros(len(rir))
env_peaks[peaks] = rir[peaks]

z = len(rir) / len(rir[peaks])

ir_env = interpolation.zoom(rir[peaks], z)
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
plt.plot(t, rir)
plt.plot(t, ir_env)
plt.plot(t, normalized_synthetic_ir, alpha = 0.4)
plt.show()

# espectrograma



f, t, Sxx = sig.spectrogram(rir, fs)
rir_dB = 20*np.log10(Sxx/2e-5)

f, t, synt_Sxx = sig.spectrogram(normalized_synthetic_ir, fs)
synth_dB = 20*np.log10(synt_Sxx/2e-5)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.pcolormesh(t, f, rir_dB, shading='gouraud')
ax2.pcolormesh(t, f, synth_dB, shading='gouraud')

plt.show()


# exportar archivos

sf.write('RIRs/synthetic_ir.wav', normalized_synthetic_ir, fs)