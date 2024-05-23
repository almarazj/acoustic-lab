from tkinter import filedialog
from pydub import AudioSegment, effects  
import soundfile as sf
import pyloudnorm as pyln

file_path = filedialog.askopenfilename(title = "Select Audio File", filetypes = [("WAV files", "*.wav")])
data, fs = sf.read(file_path) 
meter = pyln.Meter(fs)
loudness = meter.integrated_loudness(data)
loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -12.0)
sf.write('./output.wav', loudness_normalized_audio, fs)


