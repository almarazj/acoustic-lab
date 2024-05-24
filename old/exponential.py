
import numpy as np
import matplotlib.pyplot as plt
def generate_exponential_impulse_train(total_time, initial_interval, decay_rate, amplitude, sample_rate):
    """
    Generate an impulse train signal with exponentially varying pulse density.
    
    Parameters:
    total_time (float): Total time duration of the signal in seconds.
    initial_interval (float): Initial interval between impulses in seconds.
    decay_rate (float): Decay rate for the exponential function controlling the interval.
    amplitude (float): Amplitude of the impulses.
    sample_rate (float): Sampling rate in samples per second.
    
    Returns:
    np.ndarray: Array representing the impulse train signal.
    np.ndarray: Time array for the signal.
    """
    t = np.arange(0, total_time, 1/sample_rate)
    signal = np.zeros_like(t)
    
    current_time = 0
    while current_time < total_time:
        index = int(current_time * sample_rate + np.random.rand() * 250 * np.exp(-decay_rate * current_time))
        if index < len(signal):
            signal[index] = 2 * round(np.random.rand()) - 1
        current_time += initial_interval * np.exp(-decay_rate * current_time)
    
    return signal, t

# Parameters for the impulse train
total_time = 1.0       # Total time duration in seconds
initial_interval = 0.1 # Initial interval between impulses in seconds
decay_rate = 2.0       # Decay rate for the exponential function
amplitude = 1.0        # Amplitude of the impulses
sample_rate = 1000.0   # Sampling rate in samples per second

# Generate the exponentially varying impulse train
impulse_train, time_array = generate_exponential_impulse_train(total_time, initial_interval, decay_rate, amplitude, sample_rate)

# Plot the impulse train
plt.figure(figsize=(10, 4))
plt.stem(time_array, impulse_train)
plt.title("Exponentially Varying Impulse Train")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()