import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

# Load WAV file
rate, data = wav.read(
    'results/test25382-58_25382.58_20250611_162219/audio.wav.wav')

# If stereo, take one channel
if len(data.shape) > 1:
    data = data[:, 0]

# Create spectrogram
f, t, Sxx = signal.spectrogram(data, fs=rate, nperseg=1024, noverlap=512)

# Plot and save spectrogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power [dB]')
plt.tight_layout()
plt.savefig('spectrogram.png')
plt.close()

# Power spectrum (average over time), and save
power_spectrum = np.mean(Sxx, axis=1)
plt.figure(figsize=(12, 4))
plt.plot(f, 10 * np.log10(power_spectrum))
plt.title('Average Power Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [dB]')
plt.grid()
plt.tight_layout()
plt.savefig('power_spectrum.png')
plt.close()
