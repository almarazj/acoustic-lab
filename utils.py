import soundfile as sf
import os
import numpy as np

def load_ir(path, filename):
    raw_rir, fs = sf.read(os.path.join(path, filename))

    raw_duration = int(len(raw_rir) / fs)

    raw_t = np.linspace(0, raw_duration, int(fs*raw_duration))

    t = raw_t[0:int(fs*raw_duration)-np.argmax(np.abs(raw_rir))-1]
    rir = raw_rir[np.argmax(np.abs(raw_rir)):-1]

    duration = int(len(rir) / fs)
    return t, rir, fs, duration

def generate_synthetic_rir(t, rir, fs, duration):

    
    noise = np.random.normal(0, 1, int(fs*duration))
