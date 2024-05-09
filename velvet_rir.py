import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import scipy.signal as sig

# Functions

def read_audio_files():
    path = os.path.abspath(os.getcwd())
    audio_path= os.path.join(path,'audio-files')
    filenames = os.listdir(audio_path)
    audio_data = []
    for file in filenames:
        file_path = os.path.join(audio_path, file)
        data, fs = sf.read(file_path)
        audio_data.append((data, fs))
    return audio_data

def velvetNoise(duration, fs, f, ensureLast):
    # velvet(N, f, Fs)
    # INPUT
    #     N : size of signal to generate
    #     f : frequency (pulse density) of VN sequence
    #     Fs : Samplerate
    #     ensureLast : (optional) boolean. If pulse period doesn't divide
    #             evenly into N, force the last pulse to occur within the
    #             last window regardless of its size
    
    # OUTPUT
    #     Y : output signal, length N
    
    # freq (pulse density) in Hz
    N = int(duration * fs)
    T = fs/f        # pulse period
    nP = int(np.ceil(N/T))  # number of pulses to generate
    Y = np.zeros(N) # output signal
    
    # calc pulse location (Välimäki, et al, Eq. 2 & 3)
    for m in range(nP-1):                                 # m needs to start at 0
        p_idx = round((m*T) + np.random.rand()*(T-1))       # k_m, location of pulse within this pulse period
        if p_idx <= N:                                      # make sure the last pulse is in bounds (in case nP was fractional)
            Y[p_idx+1] = 2 * round(np.random.rand()) - 1    # value of pulse: 1 or -1
                                                            # p_idx+1: bump from 0- to 1-based indexing
        
        elif ensureLast == 1:
            p_idx = round((m*T) + np.random.rand()*(T-1-N%T))
            Y[p_idx+1] = 2 * round(np.random.rand()) - 1 
            print('forcing last pulse within bounds')
        
    return Y

def gaussianNoise(duration, fs):
    noise = np.random.normal(0, 0.3, int(duration * fs))
    return noise

def envelope(t, rt, A, M):
    b = 3 / (rt*np.log(np.e))
    return A * np.exp(-b * t) + M

# def convolve(audio_data):
#     data = audio_data[0]
#     fs = audio_data[1]
#     convolved_audio = np.convolve(data, rir)

duration = 3        # seconds
f = 2000           # pulse density
rt = 0.2            # reverberation time in seconds

# Read anechoic audio files
audio_data = read_audio_files()
audio = audio_data[2]
# for audio in audio_data:
data = audio[0]
fs = audio[1]
t = np.arange(0, duration, 1/fs)
vnoise = velvetNoise(duration, fs, f, 1)
gnoise = gaussianNoise(duration, fs)
env = envelope(t, rt, 1, 0.001)
v_rir = vnoise * env
g_rir = gnoise * env
audio_vn = sig.convolve(data, v_rir)
norm_audio_vn = audio_vn * (np.max(data)/np.max(audio_vn))
audio_gn = sig.convolve(data, g_rir)
norm_audio_gn = audio_gn * (np.max(data)/np.max(audio_gn))


plt.subplot(2,2,1)
plt.plot(t, v_rir)
plt.xlabel('time [s]')
plt.ylabel('amplitude')

plt.subplot(2,2,2)
plt.plot(t, g_rir)
plt.xlabel('time [s]')
plt.ylabel('amplitude')

plt.subplot(2,2,3)
plt.plot(norm_audio_vn)
plt.xlabel('time [s]')
plt.ylabel('amplitude')

plt.subplot(2,2,4)
plt.plot(norm_audio_gn)
plt.xlabel('time [s]')
plt.ylabel('amplitude')

plt.show()
sf.write('velvet.wav', v_rir, fs)
sf.write('gaussian.wav', g_rir, fs)

sf.write('drums velvet.wav', norm_audio_vn, fs)
sf.write('drums gaussian.wav', norm_audio_gn, fs)