import scipy.signal as sig
from tkinter import filedialog
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import librosa

file_path = filedialog.askopenfilename(title = "Select Audio File", filetypes = [("WAV files", "*.wav")])
x, fs = sf.read(file_path)

S = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=128,
                                    fmax=16000)
fig, ax = plt.subplots(figsize=(5,3))
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=fs,
                         fmax=16000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB', ticks=[0, -20, -40, -60, -80])
plt.yticks([512, 1024, 2048, 4096, 8192], ['0.5', '1', '2', '4', '8'])
plt.ylabel('Frecuency [kHz]')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.subplots_adjust(left=0.11, right=0.995, top=0.98, bottom=0.15)
plt.show()

# frequencies, times, spectrogram = sig.spectrogram(x, fs, scaling='spectrum')
# s = librosa.feature.melspectrogram(y=x, sr=fs)
# plt.pcolormesh(times, frequencies, 10*np.log(s), cmap='magma', vmin='-200')
# plt.colorbar()
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()



