import soundfile as sf
import os
import numpy as np
from scipy import signal
from scipy import ndimage

def load_rirs_path():
    path = os.path.abspath(os.getcwd())
    irs_path= os.path.join(path,'audio-files')
    filenames = [f for f in os.listdir(irs_path) if f.endswith('.wav')]
    filenames.sort()
    return irs_path, filenames

def load_ir(path, filename):
    raw_rir, fs = sf.read(os.path.join(path, filename))

    raw_t = np.linspace(0, len(raw_rir), len(raw_rir))

    t = raw_t[0:len(raw_rir)-np.argmax(np.abs(raw_rir))-1] / fs
    rir = raw_rir[np.argmax(np.abs(raw_rir)):-1]

    duration = len(rir) / fs
    return t, rir, fs, duration

def generate_synthetic_rir(rir):

    peaks, _ = signal.find_peaks(np.abs(rir))

    env_peaks = np.zeros(len(rir))
    env_peaks[peaks] = np.abs(rir)[peaks]

    z = len(rir) / len(rir[peaks])

    ir_env = ndimage.interpolation.zoom(np.abs(rir)[peaks], z)

    # M = 0.0005         # background noise amplitude
    # A = 0.055           # impulse response amplitude
    # rt = 0.5           # reverberation time

    # # decay rate
    # b = 3 / (rt * np.log(np.e))    

    # gaussian noise
    noise = np.random.normal(0, 1, len(rir))

    # synthetic IR
    syntetic_ir = (noise * ir_env) 
    n = np.max(syntetic_ir) / np.max(rir)
    normalized_synthetic_ir = syntetic_ir / n
    
    return normalized_synthetic_ir

