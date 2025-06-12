#!/usr/bin/env python3
"""
accumulate_payloads.py

Scan a directory containing subdirectories with payload.txt files
and aggregate all payload texts into a single JSON file.

Usage:
    python accumulate_payloads.py /path/to/results/decoded/20250612_120950

This will read each "payload.txt" under immediate subfolders (e.g., frame_00, frame_01, ...),
collect their contents, and write "aggregate.json" in the given directory:

{
  "frame_00": "...text...",
  "frame_01": "...text...",
   ...
}
"""
import argparse
import json
import os
import sys

def collect_payloads(root_dir):
    """
    Traverse immediate subdirectories of root_dir, find payload.txt
    and return a dict mapping subdirectory name -> payload text.
    """
    results = {}
    # List entries in root_dir
    try:
        entries = sorted(os.listdir(root_dir))
    except FileNotFoundError:
        print(f"Error: directory not found: {root_dir}", file=sys.stderr)
        sys.exit(1)
    for entry in entries:
        subdir = os.path.join(root_dir, entry)
        if os.path.isdir(subdir):
            payload_path = os.path.join(subdir, 'payload.txt')
            if os.path.isfile(payload_path):
                with open(payload_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read().strip()
                results[entry] = text
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate payload.txt files into a JSON.'
    )
    parser.add_argument('decoded_dir',
                        help='Path to decoded results folder (timestamp)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON filename (default: aggregate.json in decoded_dir)')
    args = parser.parse_args()

    decoded_dir = args.decoded_dir
    data = collect_payloads(decoded_dir)
    if not data:
        print(f"No payload.txt files found under {decoded_dir}", file=sys.stderr)
        sys.exit(1)

    output_file = args.output or os.path.join(decoded_dir, 'aggregate.json')
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=2)
        print(f"Wrote aggregate JSON to {output_file}")
    except IOError as e:
        print(f"Error writing JSON file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

