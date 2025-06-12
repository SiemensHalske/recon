# tests/test_runtime.py
import sys
import time
from pathlib import Path

# damit pytest unseren Extractor findet
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import pytest

from extract_callsigns import (
    CallSignProcessor,
    YamlConfigLoader,
    RegexBuilder,
    SimpleFileReader,
    CallSignClassifier,
    JsonResultSaver
)

@pytest.mark.timeout(2)  # optional: schlägt pytest ab 2 s fehl
def test_overall_runtime(tmp_path):
    # 1) Prefix-Datei anlegen
    prefixes = tmp_path / "prefixes.yaml"
    prefixes.write_text("""
icao_prefixes:
  - AA
itu_prefixes: []
military_prefixes: []
""".strip())

    # 2) Test-Verzeichnis mit ein paar kleinen Dateien füllen
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Erstelle 10 Dateien mit je 1000 Tokens
    for i in range(10):
        f = data_dir / f"file_{i}.log"
        # Wechselnde Callsigns, damit es etwas Arbeit gibt
        content = "\n".join([f"AA{j:03d}" for j in range(1000)])
        f.write_text(content)

    # 3) Processor initialisieren
    processor = CallSignProcessor(
        config_loader=YamlConfigLoader(),
        regex_builder=RegexBuilder(),
        file_reader=SimpleFileReader(),
        classifier_cls=CallSignClassifier,
        result_saver=JsonResultSaver()
    )

    # 4) Laufzeit messen
    start = time.perf_counter()
    processor.process(
        directory=data_dir,
        config_path=prefixes,
        min_count=1,
        show_other=False
    )
    elapsed = time.perf_counter() - start

    # 5) Assertion: z.B. unter 1 Sekunde
    assert elapsed < 1.0, f"CallSignExtractor zu langsam: {elapsed:.2f}s"
