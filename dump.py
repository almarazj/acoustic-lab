# espectrograma
win_size = 256
f, t, rir_spec = sig.stft(rir,
                            fs=fs, 
                            nperseg=win_size, 
                            padded=True)

rir_spec_dB = 20*np.log10(np.abs(rir_spec)/2e-5)

_, _, synt_rir_spec = sig.stft(norm_synt_ir,
                                fs=fs,
                                nperseg=win_size,
                                padded=True)

synt_rir_spec_dB = 20*np.log10(np.abs(synt_rir_spec)/2e-5)

frequency_response = median_filter(np.real(rir_spec_dB)[:,0], 25, mode='reflect')

spec_diff = median_filter(np.real(synt_rir_spec_dB)[:,0], 25, mode='reflect') - frequency_response

filt_synt_rir_spec_dB = synt_rir_spec_dB
for i in range(np.shape(synt_rir_spec_dB)[1]):
    filt_synt_rir_spec_dB[:, i] = synt_rir_spec_dB[:, i] - spec_diff

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#norm = mpl.colors.Normalize(vmin=-1, vmax=1)
mesh1 = ax1.pcolormesh(t, f, rir_spec_dB, shading='gouraud')
ax1.set_ylim(20,20000)
mesh2 = ax2.pcolormesh(t, f, synt_rir_spec_dB, shading='gouraud')
ax2.set_ylim(20,20000)
mesh3 = ax3.pcolormesh(t, f, filt_synt_rir_spec_dB, shading='gouraud')
ax3.set_ylim(20,20000)
fig.colorbar(mesh1)
fig.colorbar(mesh2)
fig.colorbar(mesh3)
plt.show()