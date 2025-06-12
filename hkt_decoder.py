#!/usr/bin/env python3
"""
stanag4285_detect_rich.py

Detect and extract multiple STANAG-4285 frames in a WAV file:
- Preamble detection via correlation
- Header demodulation (BPSK)
- Save full frame symbols and header
- Payload decoding: descramble, deinterleave, Viterbi
- 6-bit text unpacking
- Support multiple-frame detection with limited parallelism
- Print summary table at end
- Save plots + results in results/decoded/YYYYMMDD_HHMMSS
"""
# Limit BLAS/OpenMP threads to avoid CPU/RAM overuse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Use non-GUI backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

import argparse
import datetime
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, decimate, resample_poly, convolve, find_peaks

from rich.console import Console
from rich.table import Table

# === CONFIG ===
TARGET_FS      = 48000
SYMBOL_RATE    = 2400.0
CARRIER_FREQ   = 1800.0
BP_LO          = 600.0
BP_HI          = 3000.0
CORR_THRESHOLD = 1e4
PREAMBLE_LEN   = 80
HEADER_DEFAULT = 64
FRAME_DEFAULT  = 1024
DEFAULT_WORKERS = 2

# PN preamble sequence
PN_BITS = np.array([
    +1, +1, -1, +1, -1, +1, -1, -1, +1, +1,
    +1, -1, -1, +1, -1, +1, +1, -1, -1, -1,
    -1, +1, -1, -1, -1, +1, +1, +1, -1, -1,
    +1
], dtype=float)
PREAMBLE = np.concatenate([PN_BITS, PN_BITS, PN_BITS[:18]])
PREAMBLE_CORR = PREAMBLE[::-1]

# Scrambler config
SCRAMBLER_POLY = (9, 4)
SCRAMBLER_SEED = np.ones(9, dtype=int)

# Convolutional code (rate 1/2, K=7)
from commpy.channelcoding.convcode import Trellis
from commpy.channelcoding import viterbi_decode

trellis = Trellis(memory=np.array([6]), g_matrix=np.array([[0o171, 0o133]]))

console = Console()

# === SIGNAL PROCESSING HELPERS ===

def bandpass_filter(x, fs, lo, hi, order=6):
    nyq = fs / 2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, x)


def to_complex_baseband(x, fs, fc):
    xa = hilbert(x)
    t = np.arange(len(x)) / fs
    return xa * np.exp(-2j * np.pi * fc * t)


def detect_preamble(baseband, fs, sr):
    decim = int(fs // sr)
    if not np.isclose(fs / decim, sr):
        raise ValueError("fs must be integer multiple of symbol_rate")
    sym = decimate(baseband, decim, zero_phase=True)
    corr = np.abs(convolve(np.real(sym), PREAMBLE_CORR, mode='valid'))
    peaks, _ = find_peaks(corr, height=CORR_THRESHOLD, distance=FRAME_DEFAULT)
    return corr, sym, peaks


def bpsk_demod(symbols):
    return (np.real(symbols) > 0).astype(int)


def descramble(bits):
    state = SCRAMBLER_SEED.copy()
    out = np.zeros_like(bits)
    for i, b in enumerate(bits):
        fb = state[-SCRAMBLER_POLY[0]] ^ state[-SCRAMBLER_POLY[1]]
        out[i] = b ^ fb
        state = np.concatenate(([fb], state[:-1]))
    return out


def deinterleave(bits, rows=12):
    if len(bits) % rows != 0:
        return bits
    cols = len(bits) // rows
    mat = bits.reshape((cols, rows))
    return mat.T.flatten()


def bits_to_bytes(bits):
    return np.packbits(bits).tobytes()

# === DECODING ===

def decode_payload(sym, header_len):
    bits = bpsk_demod(sym)
    start = PREAMBLE_LEN + header_len
    coded = bits[start:start + 2*204]
    if len(coded) != 2*204:
        return b''
    unc = descramble(coded)
    deint = deinterleave(unc)
    llr = 1 - 2*deint
    dec = viterbi_decode(llr, trellis, tb_depth=32, decoding_type='soft')
    return bits_to_bytes(dec.astype(int))


def unpack6_text(payload):
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    text = ''
    for i in range(len(bits) // 6):
        val = int(''.join(str(b) for b in bits[i*6:(i+1)*6]), 2)
        if 1 <= val <= 26:
            text += chr(ord('A') + val - 1)
        elif val == 27:
            text += ' '
        elif 28 <= val <= 37:
            text += chr(ord('0') + val - 28)
        else:
            text += '.'
    return text

# === PLOTTING ===

def save_correlation_plot(corr, out, title='Correlation'):
    plt.figure(figsize=(10,4))
    plt.plot(corr)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'correlation.png'))
    plt.close()


def save_constellation_plot(sym, out, n=500):
    plt.figure(figsize=(5,5))
    plt.plot(np.real(sym[:n]), np.imag(sym[:n]), '.', alpha=0.5)
    plt.title('Constellation')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'constellation.png'))
    plt.close()

# === FRAME PROCESSING ===

def process_frame(i, idx, sym, corr, header_len, frame_len, out_base):
    out = os.path.join(out_base, f'frame_{i:02d}')
    os.makedirs(out, exist_ok=True)
    fsym = sym[idx:idx+frame_len]
    np.save(os.path.join(out, 'frame_symbols.npy'), fsym)
    # header
    hb = None
    hs, he = idx+PREAMBLE_LEN, idx+PREAMBLE_LEN+header_len
    if he <= len(sym):
        hb = bpsk_demod(sym[hs:he])
        hb_bytes = [int(''.join(str(b) for b in hb[j:j+8]), 2) for j in range(0, len(hb), 8)]
        open(os.path.join(out, 'header.bin'), 'wb').write(bytearray(hb_bytes))
    # payload
    payload = decode_payload(fsym, header_len)
    open(os.path.join(out, 'payload.bin'), 'wb').write(payload)
    text6 = unpack6_text(payload)
    open(os.path.join(out, 'payload.txt'), 'w').write(text6)
    # plots
    save_correlation_plot(corr, out, title=f'Frame {i}')
    save_constellation_plot(sym, out)
    time_s = idx / SYMBOL_RATE
    return (i, time_s, idx, text6)

# === MAIN DETECTION ===

def run_detection(wav_path, header_len, frame_len, workers):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_base = os.path.join('results', 'decoded', ts)
    os.makedirs(out_base, exist_ok=True)

    fs, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    filtered = bandpass_filter(audio.astype(float), fs, BP_LO, BP_HI)
    if fs != TARGET_FS:
        filtered = resample_poly(filtered, TARGET_FS, fs)
        fs = TARGET_FS
    bb = to_complex_baseband(filtered, fs, CARRIER_FREQ)

    corr, sym, peaks = detect_preamble(bb, fs, SYMBOL_RATE)
    console.print(f"Found {len(peaks)} frame(s)")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_frame, i, idx, sym, corr, header_len, frame_len, out_base)
                   for i, idx in enumerate(peaks)]
        for f in as_completed(futures):
            results.append(f.result())

    # Summary
    table = Table(title='Decoded Frames')
    table.add_column('Frame', style='cyan')
    table.add_column('Time(s)', style='magenta')
    table.add_column('Index', style='green')
    table.add_column('Text6', style='white')
    for i, time_s, idx, text6 in sorted(results):
        table.add_row(str(i), f'{time_s:.3f}', str(idx), text6)
    console.print(table)

# === CLI ===

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input_wav')
    p.add_argument('--header-len', type=int, default=HEADER_DEFAULT)
    p.add_argument('--frame-len', type=int, default=FRAME_DEFAULT)
    p.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                   help='Max parallel threads')
    return p.parse_args()


def main():
    args = parse_args()
    run_detection(args.input_wav, args.header_len, args.frame_len, args.workers)

if __name__ == '__main__':
    main()

