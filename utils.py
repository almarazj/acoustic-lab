import soundfile as sf
import os
import numpy as np
from scipy import signal
from scipy import ndimage
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
    plt.plot(ir_env)
    plt.show()
    # M = 0.0005         # background noise amplitude
    # A = 0.055           # impulse response amplitude
    # rt = 0.5           # reverberation time

    # # decay rate
    # b = 3 / (rt * np.log(np.e))    

    # gaussian noise
    noise = np.random.normal(0, 1, len(rir))
    plt.plot(noise)
    plt.show()
    # synthetic IR
    syntetic_ir = (noise * ir_env) 
    n = np.max(syntetic_ir) / np.max(rir)
    normalized_synthetic_ir = syntetic_ir / n
    plt.plot(normalized_synthetic_ir)
    plt.show()

    return normalized_synthetic_ir

def try_stft(rir, hop, fs):
    SFT = signal.ShortTimeFFT(rir, hop=hop, fs=fs)
    spec = SFT.stft(rir)

    fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit

    t_lo, t_hi = SFT.extent(len(rir))[:2]  # time range of plot
    ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Gaussian window, " +
                rf"$\sigma_t={8*SFT.T}\,$s)")
    ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(len(rir))} slices, " +
                rf"$\Delta t = {SFT.delta_t:g}\,$s)",
            ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
                rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
            xlim=(t_lo, t_hi))

    im1 = ax1.imshow(abs(spec), origin='lower', aspect='auto',
                    extent=SFT.extent(len(rir)), cmap='viridis')
    fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

    # Shade areas where window slices stick out to the side:
    for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
                    (SFT.upper_border_begin(len(rir))[0] * SFT.T, t_hi)]:
        ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
    for t_ in [0, len(rir) * SFT.T]:  # mark signal borders with vertical line:
        ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
    ax1.legend()
    fig1.tight_layout()
    plt.show()

    print(SFT.invertible)

    x1 = SFT.istft(spec, k1=len(rir))
    sf.write('audio-files/x1.wav', x1, fs)


