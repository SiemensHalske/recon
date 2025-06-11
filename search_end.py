import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import scipy.signal as signal

# Load WAV file
rate, data = wav.read('results/test25382-58_25382.58_20250611_162219/audio.wav.wav')

# If stereo, take one channel
if len(data.shape) > 1:
    data = data[:, 0]

# Normalize to [-1, 1]
data = data / np.max(np.abs(data))

# Compute analytic signal (Hilbert transform) to get instantaneous amplitude and phase
analytic_signal = signal.hilbert(data)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))

# Compute instantaneous phase difference (derivative)
phase_diff = np.diff(instantaneous_phase)

# Smooth amplitude envelope (optional)
amplitude_envelope_smooth = signal.medfilt(amplitude_envelope, kernel_size=101)

# Plot amplitude envelope
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(amplitude_envelope_smooth)) / rate, amplitude_envelope_smooth)
plt.title('Instantaneous Amplitude Envelope')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.grid()
plt.tight_layout()
plt.savefig('amplitude_envelope.png')
plt.close()

# Plot phase difference
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(phase_diff)) / rate, phase_diff)
plt.title('Instantaneous Phase Difference')
plt.xlabel('Time [sec]')
plt.ylabel('Phase Difference [rad]')
plt.grid()
plt.tight_layout()
plt.savefig('phase_difference.png')
plt.close()
