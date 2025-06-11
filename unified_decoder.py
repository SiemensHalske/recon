import argparse

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

try:  # plotting is optional
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - not critical for decoding
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
BIT_RATE = SYMBOL_RATE * 2

PREAMBLE_BITS = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 1] * 8, dtype=np.uint8)


def stanag_4285_descrambler(bit_stream, seed: int = 0x1FF) -> np.ndarray:
    """Descramble a STANAG 4285 bit stream using its 9-bit LFSR."""
    lfsr = seed
    out = np.zeros_like(bit_stream)
    for i, b in enumerate(bit_stream):
        fb = ((lfsr >> 8) & 1) ^ ((lfsr >> 3) & 1)
        out[i] = b ^ fb
        lfsr = ((lfsr << 1) & 0x1FF) | (b & 1)
    return out


def moving_variance(x: np.ndarray, w: int) -> np.ndarray:
    """Fast moving variance using cumulative sums."""
    cs1 = np.concatenate(([0], np.cumsum(x)))
    cs2 = np.concatenate(([0], np.cumsum(x * x)))
    sum1 = cs1[w:] - cs1[:-w]
    sum2 = cs2[w:] - cs2[:-w]
    return sum2 / w - (sum1 / w) ** 2


class Signal:
    """Load and prepare complex baseband from a WAV file."""

    def __init__(self, filename: str):
        self.filename = filename
        self.rate: int | None = None
        self.data: np.ndarray | None = None
        self.analytic: np.ndarray | None = None

    def load(self) -> None:
        self.rate, data = wav.read(self.filename)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(float)
        data /= np.max(np.abs(data))
        self.data = data
        self.analytic = signal.hilbert(data)

    def envelope(self, target_fs: int = TARGET_ENV_FS) -> tuple[np.ndarray, int]:
        assert self.analytic is not None and self.rate is not None
        env = np.abs(self.analytic)
        dec = max(1, self.rate // target_fs)
        return env[::dec], self.rate // dec


class PreambleDetector:
    """Detect preambles by cross-correlating with a known bit pattern."""

    def __init__(self, pattern: np.ndarray, bit_rate: int, threshold: float = 0.8):
        self.pattern = np.array(pattern, dtype=np.uint8)
        self.bit_rate = bit_rate
        self.threshold = threshold

    def detect(self, bit_stream: np.ndarray) -> list[float]:
        pattern_pm = 2 * self.pattern - 1
        stream_pm = 2 * bit_stream - 1
        corr = np.correlate(stream_pm, pattern_pm, mode="valid") / len(self.pattern)
        idx = np.where(corr >= self.threshold)[0]
        return (idx / self.bit_rate).tolist()


class Stanag4285Decoder:
    """High-level decoder for STANAG 4285 bursts."""

    def __init__(self, filename: str, preamble_bits: np.ndarray | None = None):
        self.signal = Signal(filename)
        self.preamble_bits = PREAMBLE_BITS if preamble_bits is None else np.array(
            preamble_bits, dtype=np.uint8
        )

    @staticmethod
    def _qpsk_to_bits(phase: np.ndarray, rate: int) -> np.ndarray:
        sps = rate / SYMBOL_RATE
        idx = np.arange(0, len(phase), sps).astype(int)
        idx = idx[idx < len(phase)]
        symbol_phase = phase[idx]
        qpsk = np.floor(((symbol_phase + np.pi) / (np.pi / 2))) % 4
        mapping = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
        bits = []
        for sym in qpsk:
            bits.extend(mapping[int(sym)])
        return np.array(bits, dtype=np.uint8)

    def decode(self, plot: bool = False) -> list[np.ndarray]:
        self.signal.load()
        env, rate_env = self.signal.envelope()
        w = int(WINDOW_SEC * rate_env)
        amp_var = moving_variance(env, w)
        cand_times = self._detect_candidates(amp_var, rate_env)

        inst_phase = np.unwrap(np.angle(self.signal.analytic))
        bit_stream = self._qpsk_to_bits(inst_phase, self.signal.rate)
        descrambled = stanag_4285_descrambler(bit_stream)
        finder = PreambleDetector(self.preamble_bits, BIT_RATE)
        preamble_times = finder.detect(descrambled)

        if cand_times:
            preamble_times = [t for t in preamble_times if any(abs(t - c) < 0.5 for c in cand_times)]

        if not preamble_times:
            console.print("[bold red]No bursts detected.[/bold red]")
            return []

        console.print(f"[bold cyan]Detected {len(preamble_times)} burst(s).[/bold cyan]")

        bursts_bits = []
        for t in preamble_times:
            seg = self._extract_station_id_bits(t, descrambled)
            bursts_bits.append(seg)
            console.print(f"[green]Burst at {t:.3f} sec → {bits_to_ascii(seg)}[/green]")

        if plot and plt is not None:
            time_env = np.arange(len(env)) / rate_env
            plt.figure(figsize=(12, 5))
            plt.plot(time_env[w // 2 : len(amp_var) + w // 2], amp_var)
            for t in preamble_times:
                plt.axvline(t, color="red", linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude Variance")
            plt.title("Detected Bursts")
            plt.tight_layout()
            plt.show()

        return bursts_bits

    @staticmethod
    def _detect_candidates(var: np.ndarray, rate_env: int) -> list[float]:
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

    @staticmethod
    def _extract_station_id_bits(preamble_time: float, bit_stream: np.ndarray) -> np.ndarray:
        bits_to_extract = int(EXTRACT_ID_SEC * BIT_RATE)
        bit_idx = int(preamble_time * BIT_RATE)
        return bit_stream[bit_idx : bit_idx + bits_to_extract]


def bits_to_ascii(bits: np.ndarray) -> str:
    out = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= bits[i + j] << (7 - j)
        out.append(byte)
    return "".join(chr(b) if 32 <= b <= 126 else "." for b in out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode STANAG 4285 station ID from WAV")
    parser.add_argument("wavfile", help="Input WAV file")
    parser.add_argument("--plot", action="store_true", help="Show diagnostic plot")
    args = parser.parse_args()

    Stanag4285Decoder(args.wavfile).decode(plot=args.plot)
