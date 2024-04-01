import soundfile as sf
import os
import numpy as np
from scipy import signal
from scipy import ndimage
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def load_rirs_path():
    path = os.path.abspath(os.getcwd())
    irs_path= os.path.join(path,'audio-files')
    filenames = [f for f in os.listdir(irs_path) if f.endswith('.wav')]
    filenames.sort()
    return irs_path, filenames

def load_ir(path, filename):
    raw_rir, fs = sf.read(os.path.join(path, filename))

    raw_t = np.linspace(0, len(raw_rir), len(raw_rir))

    t = raw_t[0:len(raw_rir)-np.argmax(np.abs(raw_rir))-1]
    rir = raw_rir[np.argmax(np.abs(raw_rir)):-20000]

    duration = len(rir) / fs
    return t, rir, fs, duration

def monoExp(x, m, t, b):
    return m * np.exp(t * x) + b

def fit_exp_linear(t, y):
    y = np.log(y)
    K, A_log = np.polyfit(t, y, deg=1)
    A = np.exp(A_log)
    return A, K

def generate_synthetic_rir(rir, fs):
    
    xs, _ = signal.find_peaks(np.abs(rir), distance=150)
    ys = np.abs(rir)[xs]
    
    A, K = fit_exp_linear(xs, ys)
    fit_y = monoExp(xs, A, K, 0)

    # gaussian noise
    noise = np.random.normal(0, 0.45, len(rir))
    z = len(rir) / len(rir[xs])

    ir_env = ndimage.interpolation.zoom(fit_y, z)
    
    # synthetic IR
    synthetic_ir = (noise * ir_env) 
    n = np.max(synthetic_ir) / np.max(rir)
    normalized_synthetic_ir = synthetic_ir / n
    #plt.plot(normalized_synthetic_ir,alpha=0.1, label='norm')
    # plt.plot(synthetic_ir,alpha=0.4, label='synth')
    # plt.plot(rir, label='original')
    # plt.plot(ir_env)
    # plt.legend()
    # plt.show()

    return normalized_synthetic_ir

def low_pass(rir, fs, freq, order):
    b, a = signal.butter(order, freq, fs=fs, btype='low', analog=False)
    y_low = signal.lfilter(b, a, rir)
    
    # w, h = signal.freqz(b, a, fs=fs, worN=8000)
    # plt.semilogx(w, np.abs(h))
    # plt.title('Low Pass filter')
    # plt.show()
    return y_low

def hi_pass(rir, fs, freq, order):
    b, a = signal.butter(order, freq, fs=fs, btype='high', analog=False)
    y_high = signal.filtfilt(b, a, rir)
    
    # w, h = signal.freqz(b, a, fs=fs, worN=8000)
    # plt.semilogx(w, np.abs(h))
    # plt.title('High Pass filter')
    # plt.show()
    return y_high

def split_ir(rir, fs, freq, order=6):
    det_rir = low_pass(rir, fs, freq, order)
    stoc_rir = hi_pass(rir, fs, freq, order)
    
    # plt.subplot(2,1,1)
    # plt.plot(det_rir)
    # plt.title('Deterministic rir')
    
    # plt.subplot(2,1,2)
    # plt.plot(stoc_rir)
    # plt.title('Stochastic rir')
    
    # plt.show()
    return det_rir, stoc_rir
    
def filter_synthetic_rir(synt_rir, fs, freq, order=6):
    stoc_rir = hi_pass(synt_rir, fs, freq, order)
    filtered_synthetic_rir_1 = low_pass(stoc_rir, fs, 6000, 1)
    filtered_synthetic_rir_2 =low_pass(filtered_synthetic_rir_1, fs, 14000, order)
    return filtered_synthetic_rir_2


def stft(rir, hop, fs):
    SFT = signal.ShortTimeFFT(rir, hop=hop, fs=fs)
    spec = SFT.stft(rir)
    return spec
    
def istft(spec, rir, hop, fs):
    SFT = signal.ShortTimeFFT(rir, hop=hop, fs=fs)
    x1 = SFT.istft(spec, k1=len(rir))
    return x1


