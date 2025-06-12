#!/usr/bin/env python3
import numpy as np
from commpy.channelcoding.convcode import Trellis, viterbi_decode

# === CONFIG ===
HEADER_LEN   = 128    # symbols in your header
PREAMBLE_LEN =  80    # always 80 symbols
SCRAMBLER_POLY = (9, 4)
SCRAMBLER_SEED = np.ones(9, dtype=int)

# Convolutional code: rate=1/2, constraint K=7, octal (171,133)
trellis = Trellis(memory=np.array([6]), g_matrix=np.array([[0o171, 0o133]]))

def descramble(bits):
    state = SCRAMBLER_SEED.copy()
    out = np.zeros_like(bits)
    for i, b in enumerate(bits):
        fb = state[-SCRAMBLER_POLY[0]] ^ state[-SCRAMBLER_POLY[1]]
        out[i] = b ^ fb
        state = np.concatenate(([fb], state[:-1]))
    return out

def deinterleave(bits, rows=12):
    """Invert a block interleaver of size rows x cols."""
    cols = len(bits) // rows
    # reshape into (cols, rows), then transpose to (rows, cols)
    mat = bits.reshape((cols, rows))
    mat = mat.T
    return mat.flatten()

def hard_demod(symbols):
    return (np.real(symbols) > 0).astype(int)

def bits_to_bytes(bits):
    return np.packbits(bits).tobytes()

if __name__ == "__main__":
    # load your saved frame
    frame = np.load("results/decoded/20250612_114704/frame_symbols.npy")

    # 1) Hard-decision BPSK
    bits = hard_demod(frame)

    # 2) Strip preamble + header → get 2*204 = 408 coded bits
    coded = bits[PREAMBLE_LEN + HEADER_LEN : PREAMBLE_LEN + HEADER_LEN + 408]

    # 3) Descramble
    uncoded = descramble(coded)

    # 4) Deinterleave (12 rows x 34 cols)
    deint = deinterleave(uncoded, rows=12)

    # 5) Viterbi decode (soft-input: LLRs = ±1)
    llr = 1 - 2*deint
    decoded = viterbi_decode(llr, trellis, tb_depth=32, decoding_type="soft")
    decoded = decoded.astype(int)

    # 6) Pack bits into bytes (204 info bits → 26 bytes)
    payload = bits_to_bytes(decoded)

    # 7) Save
    out = "results/decoded/20250612_114704/payload.bin"
    with open(out, "wb") as f:
        f.write(payload)

    print(f"Wrote payload.bin ({len(payload)} bytes)")

