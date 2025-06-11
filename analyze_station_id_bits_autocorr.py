import numpy as np
import glob
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

# === Parameters ===
BITS_PER_SECOND = 2400 * 2
EXTRACT_ID_SEC = 2.0
TOTAL_BITS = int(BITS_PER_SECOND * EXTRACT_ID_SEC)

# max lag to compute autocorrelation (in bits) → adjust as needed!
MAX_LAG = 2000

# === Load station_id_*.bin files ===
console.print("[bold cyan]Loading station_id_*.bin files…[/bold cyan]")
files = sorted(glob.glob('station_id_*.bin'))
if not files:
    console.print(
        "[bold red]ERROR: No station_id_*.bin files found![/bold red]")
    exit(1)

console.print(f"[green]Found {len(files)} bursts:[/green]")
for f in files:
    console.print(f" - {f}")

# === Concatenate all bits ===
console.print("[bold cyan]Concatenating bits…[/bold cyan]")
bit_stream = []
for f in files:
    bits = np.fromfile(f, dtype=np.uint8)
    if len(bits) < TOTAL_BITS:
        bits = np.pad(bits, (0, TOTAL_BITS - len(bits)))
    elif len(bits) > TOTAL_BITS:
        bits = bits[:TOTAL_BITS]
    bit_stream.extend(bits)

bit_stream = np.array(bit_stream)
console.print(f"[bold green]Total bits:[/bold green] {len(bit_stream)}")

# === Convert bits to -1 / +1 for autocorrelation
bit_stream_pm = 2 * bit_stream - 1  # 0 → -1, 1 → +1

# === Compute autocorrelation
console.print("[bold cyan]Computing autocorrelation…[/bold cyan]")
autocorr_full = np.correlate(bit_stream_pm, bit_stream_pm, mode='full')
autocorr = autocorr_full[len(autocorr_full) //
                         2:len(autocorr_full)//2+MAX_LAG].astype(float)


# === Normalize
autocorr /= np.max(np.abs(autocorr))


console.print("[bold cyan]Detecting peaks in autocorrelation…[/bold cyan]")

# — Find peaks —
# Height threshold → adjust if needed
PEAK_HEIGHT = 0.3
# Minimum distance between peaks → typical STANAG frame >200 bits
MIN_DISTANCE = 200

peaks, properties = find_peaks(
    autocorr, height=PEAK_HEIGHT, distance=MIN_DISTANCE)

if len(peaks) == 0:
    console.print("[bold red]No significant peaks found![/bold red]")
else:
    console.print(f"[bold green]Found {len(peaks)} peaks:[/bold green]")
    for i, p in enumerate(peaks):
        console.print(
            f"Peak {i+1}: Lag = {p} bits, Height = {properties['peak_heights'][i]:.3f}")

    # — Plot with peaks marked —
    plt.figure(figsize=(12, 6))
    plt.plot(autocorr, color='purple')
    plt.plot(peaks, autocorr[peaks], "rx",
             markersize=10, label='Detected Peaks')
    plt.title('Autocorrelation of Bit Stream with Detected Peaks')
    plt.xlabel('Lag [bits]')
    plt.ylabel('Normalized Autocorrelation')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('station_id_bit_autocorr_peaks.png')
    plt.close()

    console.print("[bold green]Peak plot saved as[/bold green] "
                  "[yellow]station_id_bit_autocorr_peaks.png[/yellow].")
