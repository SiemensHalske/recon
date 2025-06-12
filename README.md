# Recon Tools

Recon Tools is a suite of Python utilities for collecting and processing
STANAG 4285 high-frequency (HF) radio transmissions.  The scripts are
written with disciplined operations in mind: audio can be captured from
remote SDRs, bursts can be demodulated and decoded, and resulting
payloads can be parsed, aggregated and analyzed.  The project is
intended for hobbyist signal monitoring and research purposes.

## Table of Contents

1. [Capabilities](#capabilities)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Workflow](#detailed-workflow)
5. [Repository Layout](#repository-layout)
6. [Tests](#tests)
7. [Contributing](#contributing)
8. [Security Notes](#security-notes)
9. [License](#license)

## Capabilities

### Signal Acquisition
- **`container.py`** handles scheduled recordings from KiwiSDR/WebSDR
  instances.  Frequency lists and recording lengths are configurable.
- Recorded WAV files are organised under `results/<name>_<freq>_<timestamp>`.

### Demodulation and Decoding
- **`hkt_decoder.py`** and **`unified_decoder.py`** implement burst
detection, QPSK demodulation, descrambling and frame decoding for
STANAG 4285 signals.
- **`decode.py`** provides an alternate path with extensive amplitude
plots for offline examination of poor signal conditions.
- Decoders emit JSON summaries alongside optional diagnostic images.

### Payload Processing
- **`accumulate_payloads.py`** gathers individual `payload.txt`
  outputs into a single JSON file for bulk analysis.
- **`per_pos_ent.py`** calculates per-position entropy over aggregated
  payloads to assist in reverse engineering message formats.

### Call Sign Extraction
- **`extract_callsigns.py`** scans text logs for call sign candidates
  using prefix lists from a `prefixes.yaml` file.  The configuration
  file must be provided; otherwise processing will halt with an error.
  The script can process directories in parallel and generates a JSON
  report of likely call signs.

## Installation

Recon Tools requires Python 3.8 or newer.  Install the dependencies
using `pip`:

```bash
pip install numpy scipy matplotlib rich pyyaml mdutils
```

The `kiwiclient` directory expects `kiwirecorder.py` from the
KiwiSDR client distribution.  Place the file there to enable radio
recording.

Optionally, packages such as SoX or FFmpeg can be used to inspect or
resample recorded audio, but they are not mandatory.

## Quick Start

1. **Acquire Audio**
   ```bash
   python container.py
   ```
   Edit the parameters at the top of `container.py` to select the
   desired server, frequency list and capture duration.

2. **Decode**
   ```bash
   python unified_decoder.py results/<dir>/audio.wav --outfile decode.json --plot
   ```
   The decoder will scan for STANAG bursts, demodulate them and write
   decoded ASCII strings to `decode.json`.

3. **Aggregate**
   ```bash
   python accumulate_payloads.py results/decoded/<timestamp>
   ```
   This collects all `payload.txt` files produced during decoding and
   emits a single `aggregate.json` file for further review.

4. **Analyze**
   ```bash
   python per_pos_ent.py aggregate.json
   ```
   Entropy statistics for each character position are written to
   `entropy.json`, revealing field structure or framing markers.

## Detailed Workflow

1. **Planning** – Choose HF frequencies and schedule recording times.
   The container script can loop through multiple frequencies
   automatically.
2. **Acquisition** – While recording, data is stored with timestamps for
   traceability.  Capture durations can be tuned for short test bursts or
   continuous monitoring.
3. **Burst Detection** – Decoding scripts search for STANAG 4285
   synchronization patterns and report start/stop times for each burst.
4. **Demodulation** – QPSK symbols are extracted and descrambled to
   recover the binary message stream.
5. **Frame Parsing** – Each frame is converted to 6‑bit ASCII.  Payload
   text is placed in `payload.txt` while binary dumps are saved as
   `raw.bin` for reference.
6. **Post Processing** – Aggregation and entropy analysis scripts help
   build dictionaries of fields, markers and statistical signatures.
7. **Call Sign Extraction** – Textual logs can be scanned for known
   prefix sequences to produce a final report of potential call signs.
   If `prefixes.yaml` is not present, the script falls back to a minimal
   built‑in list.

The workflow supports both ad‑hoc decoding of individual files and
structured batch processing across many captures.

## Repository Layout

```text
recon/
├── accumulate_payloads.py   # Aggregate decoded payloads
├── analysis.py              # Plotting and numeric helpers
├── container.py             # WebSDR recording automation
├── decode.py                # Decoder variant with rich plotting
├── descramble.py            # PN sequence removal routines
├── extract_callsigns.py     # Log scanning for call signs
├── hkt_decoder.py           # High performance STANAG decoder
├── per_pos_ent.py           # Entropy computation
├── unified_decoder.py       # Burst detection and decoding pipeline
├── tests/                   # Unit tests
└── README.md
```

Each script contains a docstring with invocation examples and expected
outputs.  The `tests` directory holds small audio samples and metadata
for automated verification.

## Tests

Run the unit tests with:

```bash
pytest -q
```

The test suite exercises the call sign extraction logic and verifies
performance characteristics for burst detection and file handling.

## Contributing

1. Fork the repository and create feature branches from `master`.
2. Ensure that any new scripts include basic unit tests.
3. Use clear commit messages and keep pull requests focused.
4. External contributions are reviewed for security impact before
   merging.

## Security Notes

Recon Tools is provided for educational and research use.  Depending on
your jurisdiction, monitoring certain frequencies may require a licence
or authorisation.  The maintainers do not condone misuse.  When
recording or storing potentially sensitive traffic, protect the data as
you would any other classified material: use encrypted storage and limit
access on a need‑to‑know basis.

## License

This project is released under the MIT License.  See `LICENSE` for
full terms.
