import argparse
import os

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from rich.console import Console

console = Console()

WINDOW_SEC = 0.5
START_THRESHOLD_FACTOR = 0.4
STOP_THRESHOLD_FACTOR = 0.2
HOLD_OFF_SEC = 1.0
TARGET_ENV_FS = 100
EXTRACT_ID_SEC = 2.0
SYMBOL_RATE = 2400


def stanag_4285_descrambler(bit_stream, seed=0x1FF):
    lfsr = seed
    descrambled_bits = np.zeros_like(bit_stream)
    for i in range(len(bit_stream)):
        fb_bit = ((lfsr >> 8) & 1) ^ ((lfsr >> 3) & 1)
        descrambled_bits[i] = bit_stream[i] ^ fb_bit
        lfsr = ((lfsr << 1) & 0x1FF) | (bit_stream[i] & 1)
    return descrambled_bits


def moving_variance(x, w):
    cs1 = np.concatenate(([0], np.cumsum(x)))
    cs2 = np.concatenate(([0], np.cumsum(x * x)))
    sum1 = cs1[w:] - cs1[:-w]
    sum2 = cs2[w:] - cs2[:-w]
    return sum2 / w - (sum1 / w) ** 2


def detect_preambles(var, rate_env):
    th_start = var.max() * START_THRESHOLD_FACTOR
    th_stop = var.max() * STOP_THRESHOLD_FACTOR
    min_gap = int(HOLD_OFF_SEC * rate_env)
    above = var > th_start
    start_idx = []
    in_burst = False
    last_idx = -min_gap
    for i, val in enumerate(above):
        if not in_burst and val and i - last_idx >= min_gap:
            in_burst = True
            start_idx.append(i)
        elif in_burst and var[i] < th_stop:
            in_burst = False
            last_idx = i
    return [idx / rate_env for idx in start_idx]


def extract_station_id_bits(preamble_time, bit_stream):
    bits_per_sec = SYMBOL_RATE * 2
    bits_to_extract = int(EXTRACT_ID_SEC * bits_per_sec)
    symbol_idx = int(preamble_time * SYMBOL_RATE)
    bit_idx = symbol_idx * 2
    return bit_stream[bit_idx:bit_idx + bits_to_extract]


def bits_to_ascii(bits):
    bytes_out = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= bits[i + j] << (7 - j)
        bytes_out.append(byte)
    return ''.join(chr(b) if 32 <= b <= 126 else '.' for b in bytes_out)


def decode_wav(filename, plot=False):
    rate, data = wav.read(filename)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(float)
    data /= np.max(np.abs(data))

    analytic = signal.hilbert(data)
    env_full = np.abs(analytic)

    dec = max(1, rate // TARGET_ENV_FS)
    env = env_full[::dec]
    rate_env = rate // dec

    w = int(WINDOW_SEC * rate_env)
    amp_var = moving_variance(env, w)

    preambles = detect_preambles(amp_var, rate_env)
    if not preambles:
        console.print('[bold red]No bursts detected.[/bold red]')
        return

    console.print(f'[bold cyan]Detected {len(preambles)} burst(s).[/bold cyan]')

    inst_phase = np.unwrap(np.angle(analytic))
    sps = rate / SYMBOL_RATE
    idx = np.arange(0, len(inst_phase), sps).astype(int)
    idx = idx[idx < len(inst_phase)]
    symbol_phase = inst_phase[idx]
    qpsk = np.floor(((symbol_phase + np.pi) / (np.pi / 2))) % 4

    symbol_to_bits = {
        0: [0, 0],
        1: [0, 1],
        2: [1, 1],
        3: [1, 0],
    }

    bit_stream = []
    for sym in qpsk:
        bit_stream.extend(symbol_to_bits[int(sym)])
    bit_stream = np.array(bit_stream, dtype=np.uint8)

    descrambled = stanag_4285_descrambler(bit_stream)

    bursts_bits = []
    for t in preambles:
        seg = extract_station_id_bits(t, descrambled)
        bursts_bits.append(seg)
        text = bits_to_ascii(seg)
        console.print(f'[green]Burst at {t:.3f} sec → {text}[/green]')

    if plot and plt:
        time_env = np.arange(len(env)) / rate_env
        plt.figure(figsize=(12, 5))
        plt.plot(time_env[w//2:len(amp_var)+w//2], amp_var)
        for t in preambles:
            plt.axvline(t, color='red', linestyle='--')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude Variance')
        plt.title('Detected Bursts')
        plt.tight_layout()
        plt.show()

    return bursts_bits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode STANAG 4285 station ID from WAV.')
    parser.add_argument('wavfile', help='Input WAV file')
    parser.add_argument('--plot', action='store_true', help='Show diagnostic plot')
    args = parser.parse_args()

    decode_wav(args.wavfile, plot=args.plot)
