# tests/test_runtime_perf.py
import sys
import time
from pathlib import Path

import pytest

# damit pytest unseren Extractor findet
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_callsigns import (
    CallSignProcessor,
    YamlConfigLoader,
    RegexBuilder,
    SimpleFileReader,
    CallSignClassifier,
    JsonResultSaver,
)

@pytest.mark.parametrize("file_count", [1, 10, 50])
def test_full_run_performance(tmp_path, file_count):
    # --- Setup: Erzeuge ein Test-Verzeichnis mit ein paar kleinen Log-Dateien ---
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(file_count):
        f = data_dir / f"file_{i}.log"
        # jede Datei enthält ein paar Callsigns
        f.write_text("\n".join(["AA123", "BB456", "CC789"]) + "\n")

    # Erstelle eine minimalistische prefixes.yaml
    prefixes = tmp_path / "prefixes.yaml"
    prefixes.write_text(
        "\n".join([
            "icao_prefixes:",
            "  - AA",
            "  - BB",
            "itu_prefixes: []",
            "military_prefixes: []",
        ]) + "\n"
    )

    processor = CallSignProcessor(
        config_loader=YamlConfigLoader(),
        regex_builder=RegexBuilder(),
        file_reader=SimpleFileReader(),
        classifier_cls=CallSignClassifier,
        result_saver=JsonResultSaver()
    )

    # --- Messung ---
    start = time.time()
    processor.process(
        directory=data_dir,
        config_path=prefixes,
        min_count=1,
        show_other=False
    )
    duration = time.time() - start

    # --- Assert: komplette Laufzeit unter 2 Sekunden ---
    assert duration < 2, f"Durchlauf zu langsam: {duration:.2f}s (>2s)"
