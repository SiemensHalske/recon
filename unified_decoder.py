import argparse
from dataclasses import dataclass
from typing import Iterable, List
import json
import concurrent.futures
#
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting optional
    plt = None

from rich.console import Console

console = Console()

# === Decoder configuration ===
WINDOW_SEC = 0.5
START_THRESHOLD_FACTOR = 0.4
STOP_THRESHOLD_FACTOR = 0.2
HOLD_OFF_SEC = 1.0
TARGET_ENV_FS = 100
EXTRACT_ID_SEC = 2.0
SYMBOL_RATE = 2400
BIT_RATE = SYMBOL_RATE * 2

PREAMBLE_BITS = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 1] * 8, dtype=np.uint8)


@dataclass
class Signal:
    """Load and normalize audio as analytic signal."""

    filename: str
    rate: int | None = None
    data: np.ndarray | None = None
    analytic: np.ndarray | None = None

    def load(self) -> None:
        self.rate, data = wav.read(self.filename)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(float)
        data /= np.max(np.abs(data))
        self.data = data
        self.analytic = signal.hilbert(data)

    def envelope(self, target_fs: int = TARGET_ENV_FS) -> tuple[np.ndarray, int]:
        assert self.rate is not None and self.analytic is not None
        env = np.abs(self.analytic)
        dec = max(1, self.rate // target_fs)
        return env[::dec], self.rate // dec


class BurstFinder:
    """Detect bursts via moving variance of the envelope."""

    def __init__(self, window_sec: float = WINDOW_SEC) -> None:
        self.window_sec = window_sec

    @staticmethod
    def _moving_variance(x: np.ndarray, w: int) -> np.ndarray:
        cs1 = np.concatenate(([0], np.cumsum(x)))
        cs2 = np.concatenate(([0], np.cumsum(x * x)))
        sum1 = cs1[w:] - cs1[:-w]
        sum2 = cs2[w:] - cs2[:-w]
        return sum2 / w - (sum1 / w) ** 2

    def find(self, envelope: np.ndarray, rate: int) -> List[float]:
        w = int(self.window_sec * rate)
        var = self._moving_variance(envelope, w)
        th_start = var.max() * START_THRESHOLD_FACTOR
        th_stop = var.max() * STOP_THRESHOLD_FACTOR
        min_gap = int(HOLD_OFF_SEC * rate)

        above = var > th_start
        start_idx: List[int] = []
        in_burst = False
        last_idx = -min_gap
        for i, flag in enumerate(above):
            if not in_burst and flag and i - last_idx >= min_gap:
                in_burst = True
                start_idx.append(i)
            elif in_burst and var[i] < th_stop:
                in_burst = False
                last_idx = i
        return [idx / rate for idx in start_idx]


class QpskDemodulator:
    """Convert QPSK phase to bit stream."""

    def __init__(self, symbol_rate: int = SYMBOL_RATE) -> None:
        self.symbol_rate = symbol_rate

    def demodulate(self, phase: np.ndarray, sample_rate: int) -> np.ndarray:
        sps = sample_rate / self.symbol_rate
        idx = np.arange(0, len(phase), sps).astype(int)
        idx = idx[idx < len(phase)]
        symbols = np.floor(((phase[idx] + np.pi) / (np.pi / 2))) % 4
        mapping = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
        bits: List[int] = []
        for sym in symbols:
            bits.extend(mapping[int(sym)])
        return np.array(bits, dtype=np.uint8)


class Descrambler:
    """STANAG 4285 9-bit LFSR descrambler."""

    def __init__(self, seed: int = 0x1FF) -> None:
        self.seed = seed

    def descramble(self, bit_stream: Iterable[int]) -> np.ndarray:
        lfsr = self.seed
        out = np.zeros(len(bit_stream), dtype=np.uint8)
        for i, b in enumerate(bit_stream):
            fb = ((lfsr >> 8) & 1) ^ ((lfsr >> 3) & 1)
            out[i] = b ^ fb
            lfsr = ((lfsr << 1) & 0x1FF) | (b & 1)
        return out


class PreambleDetector:
    """Detect preambles by correlating with a known bit pattern."""

    def __init__(self, pattern: np.ndarray, bit_rate: int, threshold: float = 0.8) -> None:
        self.pattern = np.array(pattern, dtype=np.uint8)
        self.bit_rate = bit_rate
        self.threshold = threshold

    def detect(self, bit_stream: np.ndarray) -> List[float]:
        pattern_pm = 2 * self.pattern - 1
        stream_pm = 2 * bit_stream - 1
        corr = np.correlate(stream_pm, pattern_pm, mode="valid") / len(self.pattern)
        idx = np.where(corr >= self.threshold)[0]
        return (idx / self.bit_rate).tolist()


class StationIdExtractor:
    """Extract bits after the preamble."""

    def __init__(self, duration_sec: float = EXTRACT_ID_SEC, bit_rate: int = BIT_RATE) -> None:
        self.duration_sec = duration_sec
        self.bit_rate = bit_rate

    def extract(self, start_time: float, bit_stream: np.ndarray) -> np.ndarray:
        start_idx = int(start_time * self.bit_rate)
        nbits = int(self.duration_sec * self.bit_rate)
        return bit_stream[start_idx : start_idx + nbits]


class Stanag4285Decoder:
    """High-level STANAG 4285 burst decoder."""

    def __init__(self, filename: str, preamble: np.ndarray = PREAMBLE_BITS) -> None:
        self.signal = Signal(filename)
        self.burst_finder = BurstFinder()
        self.demodulator = QpskDemodulator()
        self.descrambler = Descrambler()
        self.preamble_detector = PreambleDetector(preamble, BIT_RATE)
        self.id_extractor = StationIdExtractor()

    def _bits_to_ascii(self, bits: np.ndarray) -> str:
        out: List[int] = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte |= bits[i + j] << (7 - j)
            out.append(byte)
        return "".join(chr(b) if 32 <= b <= 126 else "." for b in out)

    def decode(
        self,
        plot: bool = False,
        plot_file: str = "detected_bursts.png",
        json_outfile: str = "decoded_results.json",
    ) -> List[np.ndarray]:
        self.signal.load()
        env, rate_env = self.signal.envelope()
        burst_times = self.burst_finder.find(env, rate_env)

        phase = np.unwrap(np.angle(self.signal.analytic))
        raw_bits = self.demodulator.demodulate(phase, self.signal.rate)
        bits = self.descrambler.descramble(raw_bits)
        preambles = self.preamble_detector.detect(bits)

        if burst_times:
            preambles = [t for t in preambles if any(abs(t - c) < 0.5 for c in burst_times)]

        if not preambles:
            console.print("[bold red]No bursts detected.[/bold red]")
            return []

        console.print(f"[bold cyan]Detected {len(preambles)} burst(s).[/bold cyan]")

        def process_burst(t: float) -> dict:
            seg = self.id_extractor.extract(t, bits)
            return {
                "time": t,
                "ascii": self._bits_to_ascii(seg),
            }

        with concurrent.futures.ThreadPoolExecutor() as ex:
            burst_info = list(ex.map(process_burst, preambles))

        results: List[np.ndarray] = [self.id_extractor.extract(t, bits) for t in preambles]

        with open(json_outfile, "w") as f:
            json.dump(burst_info, f, indent=2)
        console.print(f"[green]Results written to {json_outfile}[/green]")

        if plot and plt is not None:
            time_env = np.arange(len(env)) / rate_env
            var = self.burst_finder._moving_variance(env, int(self.burst_finder.window_sec * rate_env))
            plt.figure(figsize=(12, 5))
            plt.plot(time_env[int(self.burst_finder.window_sec * rate_env / 2) : int(self.burst_finder.window_sec * rate_env / 2) + len(var)], var)
            for t in preambles:
                plt.axvline(t, color="red", linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude Variance")
            plt.title("Detected Bursts")
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode STANAG 4285 station ID from WAV")
    parser.add_argument("wavfile", help="Input WAV file")
    parser.add_argument("--plot", action="store_true", help="Save diagnostic plot")
    parser.add_argument(
        "--plot-file",
        default="detected_bursts.png",
        help="Filename for plot image",
    )
    parser.add_argument(
        "--outfile",
        default="decoded_results.json",
        help="JSON file for decoded ASCII",
    )
    args = parser.parse_args()

    Stanag4285Decoder(args.wavfile).decode(
        plot=args.plot,
        plot_file=args.plot_file,
        json_outfile=args.outfile,
    )


if __name__ == "__main__":
    main()
