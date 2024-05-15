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
        if os.path.isfile(file_path) == True:
            data, fs = sf.read(file_path)
        audio_data.append((data, fs))
    return audio_data

def velvetNoise(duration, fs, f, ensureLast, amplitude):
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
    Y[0] = 50
    # calc pulse location (Välimäki, et al, Eq. 2 & 3)
    for m in range(nP-1):                                 # m needs to start at 0
        p_idx = round((m*T) + np.random.rand()*(T-1))       # k_m, location of pulse within this pulse period
        if p_idx <= N:                                      # make sure the last pulse is in bounds (in case nP was fractional)
            Y[p_idx+1] = (2 * round(np.random.rand()) - 1) * amplitude    # value of pulse: 1 or -1
                                                            # p_idx+1: bump from 0- to 1-based indexing
        elif ensureLast == 1:
            p_idx = round((m*T) + np.random.rand()*(T-1-N%T))
            Y[p_idx+1] = round(np.random.rand()) - 1 
            print('forcing last pulse within bounds')
        
    return Y

def expVelvetNoise(duration, fs, initial_interval, decay_rate):
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros_like(t)
    signal[0] = 10
    current_time = 0
    while current_time < duration:
        index = int(current_time * fs + np.random.rand() * initial_interval * fs  * np.exp(-decay_rate * current_time))
        if index < len(signal):
            signal[index] = 2 * round(np.random.rand()) - 1
        current_time += initial_interval * np.exp(-decay_rate * current_time)
    
    return signal

def gaussianNoise(duration, fs):
    Y = np.random.normal(0, 0.15, int(duration * fs))
    Y[0] = 50
    return Y

def envelope(t, rt, A, M):
    b = 3 / (rt*np.log(np.e))
    return A * np.exp(-b * t) + M

def suavizado(F,AMP,OCT):
    ampsmooth=AMP
    if OCT!=0:
        for n in range(1,len(F)):
            fsup=F[n]*pow(2,1/(2*OCT))  #calcula el corte superior del promedio
            finf=F[n]*pow(2,1/(2*OCT))  #calcula el corte inferior del promedio

            if F[-1]<=fsup:
                idxsup=len(F)-n
            else:
                idxsup=np.argmin(abs(F[n:]-fsup))   #busca el índice de fsup
                
            if F[1]<=finf:
                idxinf=np.argmin(abs(F[0:n+1]-finf))    #busca el ínfice de finf
            else:
                idxinf=0
                
            if idxsup!=idxinf+n:
                temp=pow(10,AMP[idxinf:idxsup+n-1]*0.1)
                ampsmooth[n]=10*np.log10(sum(temp)/(idxsup+n-idxinf))
    return ampsmooth

duration = 3        # seconds
amp_correction = 3.8
f = 70              # pulse density
rt = 1              # reverberation time in seconds

# real ir
ir, _ = sf.read('audio-files/ir.wav')

# Read anechoic audio files
audio_data = read_audio_files()
(data, fs) = audio_data[3]

t = np.arange(0, duration, 1/fs)
vnoise = velvetNoise(duration, fs, f, 1, amp_correction)
gnoise = gaussianNoise(duration, fs)
#exp_vnoise = expVelvetNoise(duration, fs, 0.1, 2)
env = envelope(t, rt, 1, 0.00001)

v_rir = vnoise * env
g_rir = gnoise * env

audio_vn = sig.convolve(data, v_rir)
norm_audio_vn = audio_vn * (np.max(data)/np.max(audio_vn))
audio_gn = sig.convolve(data, g_rir)
norm_audio_gn = audio_gn * (np.max(data)/np.max(audio_gn))
audio_ir = sig.convolve(data, ir)
norm_audio_ir = audio_ir * (np.max(data)/np.max(audio_ir))

vA = np.fft.fft(vnoise)
gA = np.fft.fft(gnoise)
vX = 20*np.log10(np.abs(vA))
gX = 20*np.log10(np.abs(gA))
freq = np.arange(0, len(vX), 1)
vpower_3 = suavizado(freq[20:20000], vX[20:20000], 3)
gpower_3 = suavizado(freq[20:20000], gX[20:20000], 3)


plt.semilogx(vpower_3, 'k')
plt.semilogx(gpower_3, 'r')
plt.ylim(0, 50)
plt.xlim(20,16000)
plt.xlabel('frequency [Hz]')
plt.ylabel('Level [dB]')
plt.show()
# plt.subplot(2,1,2)
# plt.semilogx(gpower_3, 'k')
# plt.ylim(0, 50)
# plt.xlim(20,16000)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('Level [dB]')

# plt.subplot(2,1,2)
# plt.plot(t, v_rir, 'k', label='Velvet noise (2000 p/s)')
# plt.ylabel('Amplitude')
# plt.xticks([])
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(t, g_rir, 'k', label='White noise')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()




#sf.write('audio-files/v90-noise.wav', vnoise, fs)3
#sf.write('audio-files/v90-drums.wav', norm_audio_vn, fs)
#sf.write('audio-files/gaussian-noise.wav', gnoise, fs)
sf.write('audio-files/v70-vocal.wav', norm_audio_vn, fs)