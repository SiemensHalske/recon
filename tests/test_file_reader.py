# tests/test_file_reader.py
import sys
from pathlib import Path

# Damit pytest unseren extractor findet
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_callsigns import SimpleFileReader

def test_file_reader_extracts_tokens(tmp_path):
    # Erstelle temporäre Log-Datei
    p = tmp_path / "sample.log"
    p.write_text("abc123\nDEF456\nFooBAR\n")

    # Lese Tokens aus
    fr = SimpleFileReader()
    tokens = fr.read_tokens(p)

    # Prüfe, dass Großschreibung und Regex greifen
    assert "ABC123" in tokens
    assert "DEF456" in tokens

    # FOOBAR ist ebenfalls ein reines Alphanum-Token (6 Buchstaben) und wird erfasst
    assert "FOOBAR" in tokens
