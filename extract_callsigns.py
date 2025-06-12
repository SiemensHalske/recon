#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CallSignExtractor — OOP, SOLID, robust und fehlertolerant

Extrahiert Call Signs aus WebSDR-STANAG-Transmissions in Textdateien.
"""

import os
import re
import argparse
import json
import datetime
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Pattern, Protocol, Iterator
from collections import Counter
import concurrent.futures

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML wird benötigt. Installiere mit 'pip install PyYAML'") from e

# -----------------------------------------------------------------------------
# Logging konfigurieren
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z"
)
logger = logging.getLogger("CallSignExtractor")

# -----------------------------------------------------------------------------
# Interfaces (Protocol) für Dependency Inversion
# -----------------------------------------------------------------------------
class ConfigLoaderInterface(Protocol):
    def load_prefixes(self, path: Path) -> Tuple[List[str], List[str], List[str]]:
        ...

class RegexBuilderInterface(Protocol):
    def build(self,
              icao: List[str],
              itu: List[str],
              military: List[str]
    ) -> Tuple[Pattern[str], Pattern[str]]:
        ...

class FileReaderInterface(Protocol):
    def read_tokens(self, filepath: Path) -> Iterator[str]:
        ...

class ClassifierInterface(Protocol):
    def classify(self, token: str) -> Optional[str]:
        ...

class ResultSaverInterface(Protocol):
    def save(self,
             directory: Path,
             likely: Dict[str,int],
             other: Dict[str,int]
    ) -> Path:
        ...

# -----------------------------------------------------------------------------
# Implementierungen
# -----------------------------------------------------------------------------
class YamlConfigLoader:
    def load_prefixes(self, path: Path) -> Tuple[List[str], List[str], List[str]]:
        if not path.is_file():
            logger.error("Config-Datei nicht gefunden: %s", path)
            raise FileNotFoundError(f"Config-Datei nicht gefunden: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        def _ensure_list(key: str) -> List[str]:
            raw = data.get(key, [])
            if not isinstance(raw, list):
                logger.warning("Erwarte Liste für '%s', erhalten %s", key, type(raw))
                return []
            return [str(x).strip().upper() for x in raw if isinstance(x, (str, int))]
        return _ensure_list("icao_prefixes"), _ensure_list("itu_prefixes"), _ensure_list("military_prefixes")

class RegexBuilder:
    def build(self,
              icao: List[str],
              itu: List[str],
              military: List[str]
    ) -> Tuple[Pattern[str], Pattern[str]]:
        valid = set(icao + itu + military)
        if not valid:
            raise ValueError("Prefix-Listen dürfen nicht leer sein")
        prefix_pat = r"^(?:" + "|".join(map(re.escape, valid)) + r")[A-Z0-9]*$"
        generic_pat = r"^[A-Z]{1,2}[A-Z0-9]{2,5}$"
        return re.compile(prefix_pat), re.compile(generic_pat)

class SimpleFileReader:
    TOKEN_REGEX = re.compile(r"\b[A-Z0-9]{2,8}\b")

    def read_tokens(self, filepath: Path) -> Iterator[str]:
        with filepath.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for token in self.TOKEN_REGEX.findall(line.strip().upper()):
                    yield token

class CallSignClassifier:
    def __init__(self, prefix_re: Pattern[str], generic_re: Pattern[str]):
        self._prefix_re = prefix_re
        self._generic_re = generic_re

    def classify(self, token: str) -> Optional[str]:
        if self._prefix_re.match(token):
            return "likely"
        if self._generic_re.match(token):
            return "other"
        return None

class JsonResultSaver:
    def save(self,
             directory: Path,
             likely: Dict[str,int],
             other: Dict[str,int]
    ) -> Path:
        now = datetime.datetime.utcnow()
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        out = directory / f"callsigns_{stamp}.json"
        payload = {
            "timestamp_utc": now.replace(microsecond=0).isoformat() + "Z",
            "input_directory": str(directory),
            "likely_callsigns": [{"callsign": k, "count": v} for k, v in sorted(likely.items(), key=lambda x: -x[1])],
            "other_callsigns": [{"callsign": k, "count": v} for k, v in sorted(other.items(), key=lambda x: -x[1])]
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Ergebnisse gespeichert: %s", out)
        return out

# -----------------------------------------------------------------------------
# Orchestrator mit Parallelisierung & Streaming
# -----------------------------------------------------------------------------
class CallSignProcessor:
    def __init__(self,
                 config_loader: ConfigLoaderInterface,
                 regex_builder: RegexBuilderInterface,
                 file_reader: FileReaderInterface,
                 classifier_cls: type,
                 result_saver: ResultSaverInterface
    ):
        self._config_loader = config_loader
        self._regex_builder = regex_builder
        self._file_reader = file_reader
        self._classifier_cls = classifier_cls
        self._result_saver = result_saver

    def process(self,
                directory: Path,
                config_path: Path,
                min_count: int,
                show_other: bool
    ) -> None:
        logger.info("Starte Verarbeitung: %s", directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Kein Verzeichnis: {directory}")

        icao, itu, mil = self._config_loader.load_prefixes(config_path)
        prefix_re, generic_re = self._regex_builder.build(icao, itu, mil)
        classifier = self._classifier_cls(prefix_re, generic_re)

        # Alle Text/Log-Dateien sammeln
        files = [Path(root)/f for root,_,fs in os.walk(directory) for f in fs if f.lower().endswith((".txt",".log"))]
        counts = Counter()

        # Parallel einlesen & token-streaming
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._file_reader.read_tokens, fp): fp for fp in files}
            for fut in concurrent.futures.as_completed(futures):
                fp = futures[fut]
                try:
                    for token in fut.result():
                        counts[token] += 1
                except Exception as e:
                    logger.warning("Fehler beim Lesen %s: %s", fp, e)

        # Klassifizieren & Filtern
        likely, other = {}, {}
        for token, cnt in counts.items():
            if cnt < min_count:
                continue
            cls = classifier.classify(token)
            if cls == "likely": likely[token] = cnt
            if cls == "other" and show_other: other[token] = cnt

        self._print_summary(likely, other, min_count, show_other)
        self._result_saver.save(directory, likely, other)
        logger.info("Verarbeitung abgeschlossen")

    def _print_summary(self, likely: Dict[str,int], other: Dict[str,int], min_count: int, show_other: bool):
        sep = '-'*60
        print(sep)
        print(f"Likely real callsigns (min_count={min_count}):")
        for cs, cnt in sorted(likely.items(), key=lambda x: -x[1]): print(f"  {cs:<8}{cnt:>5}")
        if show_other and other:
            print(sep)
            print(f"Other callsign candidates (min_count={min_count}):")
            for cs, cnt in sorted(other.items(), key=lambda x: -x[1]): print(f"  {cs:<8}{cnt:>5}")
        print(sep)

# -----------------------------------------------------------------------------
# CLI-Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract callsigns from directory.")
    parser.add_argument("directory", type=Path, help="Input-Verzeichnis")
    parser.add_argument("--prefixes", type=Path, default=Path("prefixes.yaml"), help="YAML mit Prefixes")
    parser.add_argument("--min-count", type=int, default=1, help="Minimale Häufigkeit")
    parser.add_argument("--hide-other", action="store_true", help="Andere Kandidaten nicht anzeigen")
    args = parser.parse_args()

    processor = CallSignProcessor(
        config_loader=YamlConfigLoader(),
        regex_builder=RegexBuilder(),
        file_reader=SimpleFileReader(),
        classifier_cls=CallSignClassifier,
        result_saver=JsonResultSaver()
    )
    try:
        processor.process(args.directory, args.prefixes, args.min_count, not args.hide_other)
    except Exception as e:
        logger.error("Verarbeitungsfehler: %s", e)
        exit(1)

if __name__ == "__main__":
    main()
