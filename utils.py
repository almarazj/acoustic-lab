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

def stft(rir, hop, fs):
    SFT = signal.ShortTimeFFT(rir, hop=hop, fs=fs)
    spec = SFT.stft(rir)
    return spec
    
def istft(spec, rir, hop, fs):
    SFT = signal.ShortTimeFFT(rir, hop=hop, fs=fs)
    x1 = SFT.istft(spec, k1=len(rir))
    return x1


