import numpy as np
import matplotlib.pyplot as plt
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

duration = 1
fs = 100
f = 20

N = int(duration * fs)
T = fs/f        # pulse period
nP = int(np.ceil(N/T))  # number of pulses to generate


zeros = np.zeros(N)
Y = np.zeros(N) # output signal
t = np.arange(0,duration, 1/fs)

for m in range(nP):                                 # m needs to start at 0
    p_idx = round((m*T))    # k_m, location of pulse within this pulse period
    Y[p_idx] = 1
        
Y2 = np.zeros(N) # output signal       
        
for m in range(nP):                                 # m needs to start at 0
    p_idx = round((m*T) + np.random.rand()*(T-1)) # k_m, location of pulse within this pulse period
    if p_idx <=N:
        Y2[p_idx] = 1 
    else:
        p_idx = round((m*T))
        Y[p_idx] = 1

Y3 = np.zeros(N)

for m in range(len(Y2)):
    if Y2[m] != 0:
        Y3[m] = (2 * round(np.random.rand()) - 1) 

# for m in range(nP-1):                                 # m needs to start at 0
#     p_idx = round((m*T) + np.random.rand()*(T-1))       # k_m, location of pulse within this pulse period
#     if p_idx <= N:                                      # make sure the last pulse is in bounds (in case nP was fractional)
#         Y3[p_idx+1] = (2 * round(np.random.rand()) - 1)    # value of pulse: 1 or -1
#                                                             # p_idx+1: bump from 0- to 1-based indexing
#     else:
#         p_idx = round(m*T)
#         Y3[p_idx] = round(np.random.rand()) - 1 
        
plt.subplot(3,1,1)        
plt.vlines(t, zeros, Y, 'k')
plt.plot(t,zeros, 'k')
plt.ylim([-1.1,1.1])


plt.subplot(3,1,2)
plt.vlines(t, zeros, Y2, 'k')
plt.plot(t,zeros, 'k')
plt.ylim([-1.1,1.1])


plt.subplot(3,1,3)
plt.vlines(t, zeros, Y3, 'k')
plt.plot(t,zeros, 'k')
plt.ylim([-1.1,1.1])
plt.xlabel('Time [s]')

plt.show()