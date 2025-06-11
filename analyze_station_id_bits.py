import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from rich.console import Console

console = Console()

# === Parameters ===
BITS_PER_SECOND = 2400 * 2  # QPSK → 2 bits per symbol
EXTRACT_ID_SEC = 2.0
TOTAL_BITS = int(BITS_PER_SECOND * EXTRACT_ID_SEC)

# === Load station_id_*.bin files ===
console.print("[bold cyan]Loading station_id_*.bin files…[/bold cyan]")
files = sorted(glob.glob('station_id_*.bin'))
if not files:
    console.print("[bold red]ERROR: No station_id_*.bin files found![/bold red]")
    exit(1)

console.print(f"[green]Found {len(files)} bursts:[/green]")
for f in files:
    console.print(f" - {f}")

# === Build bit matrix ===
bit_matrix = []
for f in files:
    bits = np.fromfile(f, dtype=np.uint8)
    if len(bits) < TOTAL_BITS:
        console.print(f"[yellow]WARNING: {f} has only {len(bits)} bits, padding…[/yellow]")
        bits = np.pad(bits, (0, TOTAL_BITS - len(bits)))
    elif len(bits) > TOTAL_BITS:
        bits = bits[:TOTAL_BITS]
    bit_matrix.append(bits)

bit_matrix = np.array(bit_matrix)
console.print(f"[bold green]Bit matrix shape:[/bold green] {bit_matrix.shape} (bursts × bits)")

# === Per-bit mean and entropy ===
console.print("[bold cyan]Computing per-bit mean and entropy…[/bold cyan]")
bit_mean = np.mean(bit_matrix, axis=0)

# Entropy per bit position: H = -p*log2(p) - (1-p)*log2(1-p)
epsilon = 1e-12  # to avoid log(0)
p = bit_mean
entropy = -p * np.log2(p + epsilon) - (1 - p) * np.log2(1 - p + epsilon)

# === Plot results ===
console.print("[bold cyan]Plotting results…[/bold cyan]")
plt.figure(figsize=(14, 8))

# — Bit mean plot
plt.subplot(2,1,1)
plt.plot(bit_mean, color='blue')
plt.title('Per-bit Mean (Stability)')
plt.xlabel('Bit Index')
plt.ylabel('Mean Bit Value')
plt.grid(True)

# — Entropy plot
plt.subplot(2,1,2)
plt.plot(entropy, color='red')
plt.title('Per-bit Entropy')
plt.xlabel('Bit Index')
plt.ylabel('Entropy [bits]')
plt.grid(True)

plt.tight_layout()
plt.savefig('station_id_bit_analysis.png')
plt.close()

console.print("[bold green]Done![/bold green] "
              "See [yellow]station_id_bit_analysis.png[/yellow] for results.")
