#!/usr/bin/env python3
"""
per_pos_ent.py

Compute per-position entropy and most-common character across fixed-length 6-bit ASCII
dumps in an aggregated JSON file produced by accumulate_payloads.py.

Usage:
    python per_pos_ent.py /path/to/aggregate.json [--threshold 3.5]
"""
import sys
import json
import numpy as np
from collections import Counter
import argparse

def compute_entropy(probs):
    # avoid log2(0)
    p = probs[probs > 0]
    return -np.sum(p * np.log2(p))


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-column entropy of ASCII6 payloads.'
    )
    parser.add_argument('aggregate_json', help='Path to aggregate.json')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Entropy threshold to filter columns (only show ≤ threshold)')
    parser.add_argument('--show-lowest', type=int, default=5,
                        help='Count of lowest-entropy positions to show if threshold filters all')
    args = parser.parse_args()

    try:
        data = json.load(open(args.aggregate_json, 'r', encoding='utf-8'))
    except Exception as e:
        print(f"Error loading JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # extract ascii6 strings
    texts = []
    for frame, val in data.items():
        if isinstance(val, str):
            txt = val
        elif isinstance(val, dict):
            txt = val.get('ascii6', '')
        else:
            continue
        if txt:
            texts.append(txt)
    if not texts:
        print("No ascii6 entries found in JSON.", file=sys.stderr)
        sys.exit(1)

    # report payload length distribution
    lengths = [len(t) for t in texts]
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    print(f"Payload lengths -- min: {min_len}, max: {max_len}, avg: {avg_len:.1f}")

    # pad shorter strings
    padded = [t.ljust(max_len, '.') for t in texts]
    arr = np.array([list(t) for t in padded])  # (n_frames, max_len)

    # compute all entropies
    total_frames = arr.shape[0]
    stats = []  # list of (col, entropy, most, freq)
    for col in range(arr.shape[1]):
        col_vals = arr[:, col]
        counts = Counter(col_vals)
        freq_most = counts.most_common(1)[0]
        probs = np.array(list(counts.values()), dtype=float) / total_frames
        ent = compute_entropy(probs)
        stats.append((col, ent, freq_most[0], freq_most[1]))

    # filter by threshold
    filtered = [s for s in stats if args.threshold is None or s[1] <= args.threshold]

    # header
    print("Pos\tEntropy\tMostCommon\tFreq")
    if filtered:
        for col, ent, most, freq in filtered:
            print(f"{col}\t{ent:.3f}\t'{most}'\t{freq}/{total_frames}")
    else:
        print(f"No positions with entropy ≤ {args.threshold}. Showing lowest {args.show_lowest}:\n")
        for col, ent, most, freq in sorted(stats, key=lambda x: x[1])[:args.show_lowest]:
            print(f"{col}\t{ent:.3f}\t'{most}'\t{freq}/{total_frames}")

if __name__ == '__main__':
    main()
