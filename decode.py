import os
# — limit threading to avoid 100% CPU burn —
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress

console = Console()

# === Parameters ===
WINDOW_SEC                = 0.5    # moving‐variance window
burst_end_threshold_factor = 0.2   # threshold = 20% of peak variance
burst_end_holdoff_sec      = 1.0   # must stay below threshold this long
frame_search_range         = (0.05, 0.5)   # 50–500 ms expected frame period
MAX_AUTOCORR_SEC           = 2.0   # only correlate first 2 s of each burst
TARGET_ENV_FS              = 100   # down‐sample envelope to 100 Hz

# === Station ID extraction parameters ===
EXTRACT_ID_SEC = 2.0    # How many seconds after preamble to extract
SYMBOL_RATE    = 2400   # STANAG 4285 symbol rate (for bit stream)

# === STANAG 4285 descrambler function ===
def stanag_4285_descrambler(bit_stream, seed=0x1FF):
    lfsr = seed
    descrambled_bits = np.zeros_like(bit_stream)
    for i in range(len(bit_stream)):
        fb_bit = ((lfsr >> 8) & 1) ^ ((lfsr >> 3) & 1)
        descrambled_bits[i] = bit_stream[i] ^ fb_bit
        lfsr = ((lfsr << 1) & 0x1FF) | (bit_stream[i] & 1)
    return descrambled_bits

# === Load & normalize WAV ===
console.print("[bold cyan]Loading WAV file…[/bold cyan]")
wav_file = 'results/test25382-58_25382.58_20250611_164240/audio.wav.wav'
rate, data = wav.read(wav_file)
if data.ndim > 1:
    data = data[:, 0]
data = data.astype(float) / np.max(np.abs(data))

# === Envelope & down-sampling ===
console.print("[bold cyan]Computing envelope & down-sampling…[/bold cyan]")
analytic_signal = signal.hilbert(data)
env_full = np.abs(analytic_signal)
# decimation factor to get ~TARGET_ENV_FS Hz
DEC = max(1, rate // TARGET_ENV_FS)
env = env_full[::DEC]
rate_env = rate // DEC
time_env = np.arange(len(env)) / rate_env

# === Moving variance via cumsum (O(N)) ===
console.print("[bold cyan]Computing moving variance…[/bold cyan]")
w = int(WINDOW_SEC * rate_env)
cs1 = np.concatenate(([0], np.cumsum(env)))
cs2 = np.concatenate(([0], np.cumsum(env * env)))
sum1 = cs1[w:] - cs1[:-w]
sum2 = cs2[w:] - cs2[:-w]
amp_var = sum2/w - (sum1/w)**2
time_var = (np.arange(len(amp_var)) + w/2) / rate_env

# === Load preamble detections ===
console.print("[bold cyan]Loading preamble detections…[/bold cyan]")
preamble_times = []
with open('preamble_detected.txt', 'r') as f:
    for line in f:
        if 'Preamble detected at' in line:
            t_sec = float(line.strip().split('at ')[1].split()[0])
            preamble_times.append(t_sec)

# === Symbol clock recovery ===
console.print("[bold cyan]Performing symbol clock recovery…[/bold cyan]")
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
samples_per_symbol = rate / SYMBOL_RATE
symbol_indices = np.arange(0, len(instantaneous_phase), samples_per_symbol).astype(int)
symbol_indices = symbol_indices[symbol_indices < len(instantaneous_phase)]
symbol_phase = instantaneous_phase[symbol_indices]
qpsk_symbols = np.floor(((symbol_phase + np.pi) / (np.pi / 2))) % 4

# === QPSK symbol to bits mapping ===
symbol_to_bits = {
    0: [0, 0],
    1: [0, 1],
    2: [1, 1],
    3: [1, 0],
}

bit_stream = []
for sym in qpsk_symbols:
    bit_stream.extend(symbol_to_bits[int(sym)])
bit_stream = np.array(bit_stream)

# === Descramble bits ===
console.print("[bold cyan]Descrambling bit stream…[/bold cyan]")
descrambled_bits = stanag_4285_descrambler(bit_stream)

# === Station ID extraction helper ===
def extract_station_id_bits(preamble_time, bit_stream):
    bits_per_sec = SYMBOL_RATE * 2
    bits_to_extract = int(EXTRACT_ID_SEC * bits_per_sec)
    symbol_idx = int(preamble_time * SYMBOL_RATE)
    bit_idx = symbol_idx * 2
    segment = bit_stream[bit_idx:bit_idx+bits_to_extract]
    return segment

# === Analyze bursts ===
console.print("[bold cyan]Analyzing bursts…[/bold cyan]")
burst_results = []

holdoff_samps = int(burst_end_holdoff_sec * rate_env)
lag_min = int(frame_search_range[0] * rate_env)
lag_max = int(frame_search_range[1] * rate_env)

with Progress() as progress:
    task = progress.add_task("[green]Processing bursts…", total=len(preamble_times))
    for preamble_time in preamble_times:
        # starting index in down-sampled variance
        idx0 = int(preamble_time * rate_env)
        sub_var = amp_var[idx0:]

        # — Estimate frame length via FFT autocorrelation
        max_ac_samps = int(MAX_AUTOCORR_SEC * rate_env)
        x = sub_var[:max_ac_samps]
        corr = signal.correlate(x, x, mode='full', method='fft')
        corr = corr[len(corr)//2:]
        corr[0] = 0
        pr = corr[lag_min:lag_max]
        peak_lag = np.argmax(pr) + lag_min
        frame_length_est = peak_lag / rate_env

        # — Detect burst end
        threshold = sub_var.max() * burst_end_threshold_factor
        mask = (sub_var < threshold).astype(int)
        runs = np.convolve(mask, np.ones(holdoff_samps, dtype=int), mode='valid')
        i = np.argmax(runs >= holdoff_samps)
        if runs[i] >= holdoff_samps:
            burst_end_time = time_var[idx0 + i]
        else:
            burst_end_time = time_var[-1]

        burst_results.append((preamble_time, frame_length_est, burst_end_time))

        # === Station ID extraction ===
        segment_bits = extract_station_id_bits(preamble_time, bit_stream=descrambled_bits)

        # Save raw bits to file
        id_file = f'station_id_{preamble_time:.3f}.bin'
        segment_bits.astype(np.uint8).tofile(id_file)

        # Attempt simple ASCII decode
        byte_vals = []
        for j in range(0, len(segment_bits), 8):
            byte = 0
            for k in range(8):
                if j+k < len(segment_bits):
                    byte |= segment_bits[j+k] << (7-k)
            byte_vals.append(byte)

        ascii_text = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in byte_vals)
        console.print(f'[bold yellow]Preamble at {preamble_time:.3f} sec → Station ID (ASCII guess):[/bold yellow] "{ascii_text}"')

        progress.update(task, advance=1)

# === Save burst timeline ===
console.print("[bold cyan]Saving burst_timeline.txt…[/bold cyan]")
with open('burst_timeline.txt', 'w') as f:
    for start, frame_len, end in burst_results:
        f.write(f'Burst from {start:.3f} sec to {end:.3f} sec, '
                f'Estimated frame length: {frame_len*1000:.1f} ms\n')

# === Plot combined results ===
console.print("[bold cyan]Saving combined_amplitude_bursts.png…[/bold cyan]")
plt.figure(figsize=(14, 6))
plt.plot(time_var, amp_var, alpha=0.6, label='Amplitude Variance')

for i, (start, frame_len, end) in enumerate(burst_results):
    color = 'red'
    alpha = 0.3
    plt.axvspan(start, end, color=color, alpha=alpha,
                label='Detected Burst' if i == 0 else None)
    plt.axvline(start, color='green', linestyle='--',
                label='Preamble' if i == 0 else None)

plt.title('Amplitude Variance with Detected Bursts')
plt.xlabel('Time [sec]')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('combined_amplitude_bursts.png')
plt.close()

console.print("[bold green]Done![/bold green] "
              "See [yellow]burst_timeline.txt[/yellow], "
              "[yellow]combined_amplitude_bursts.png[/yellow], "
              "and station_id_*.bin files.")

