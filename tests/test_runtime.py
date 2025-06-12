# tests/test_runtime.py
import sys
import time
from pathlib import Path

import pytest

# Damit pytest unseren Extractor findet
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_callsigns import (
    CallSignProcessor,
    YamlConfigLoader,
    RegexBuilder,
    SimpleFileReader,
    CallSignClassifier,
    JsonResultSaver
)

def test_runtime_small_dataset(tmp_path):
    """
    Testet, ob der gesamte Durchlauf auf einem kleinen Datensatz
    in unter 2 Sekunden abgeschlossen ist.
    """
    # 1) Erstelle ein kleines Test-Verzeichnis mit vielen Dateien
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Jede Datei enthält 1000 mal ein gültiges Callsign-Paar
    sample_content = "\n".join(["AA123 BB456"] * 1000)
    for i in range(50):  # 50 Dateien → insgesamt 50 000 Zeilen
        (data_dir / f"file_{i:02d}.log").write_text(sample_content)

    # 2) Initialisiere den Processor
    processor = CallSignProcessor(
        config_loader=YamlConfigLoader(),
        regex_builder=RegexBuilder(),
        file_reader=SimpleFileReader(),
        classifier_cls=CallSignClassifier,
        result_saver=JsonResultSaver()
    )

    # 3) Messen der Laufzeit
    start = time.perf_counter()
    processor.process(
        directory=data_dir,
        config_path=Path("prefixes.yaml"),
        min_count=1,
        show_other=False
    )
    duration = time.perf_counter() - start

    # 4) Assertion: darf nicht länger als 2 Sekunden dauern
    assert duration < 2.0, f"Processing took too long: {duration:.2f}s"
