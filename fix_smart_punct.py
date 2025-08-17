#!/usr/bin/env python3
import argparse
import os
import re
from datetime import datetime
from pathlib import Path

DEFAULT_EXTS = {
    ".py",
    ".txt",
    ".md",
    ".env",
    ".ini",
    ".cfg",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".sh",
    ".bat",
    ".ps1",
    ".conf",
    ".csv",
}

# Mapowanie "smart" -> ASCII
REPLACEMENTS = {
    # Pojedyncze cudzysłowy/apostrofy
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\u201b": "'",  # '
    "\u02bc": "'",  # ' (modifier letter apostrophe)
    "\u2032": "'",  # ' (prime) - uwaga: może być używany w miarach
    # Podwójne cudzysłowy
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u201f": '"',  # "
    "\u201e": '"',  # "
    "\u00ab": '"',  # "
    "\u00bb": '"',  # "
    "\u2033": '"',  # " (double prime) - jw.
    # Pauzy / myślniki
    "\u2010": "-",  # - hyphen
    "\u2011": "-",  # - non-breaking hyphen
    "\u2012": "-",  # - figure dash
    "\u2013": "-",  # - en dash
    "\u2014": "-",  # - em dash
    "\u2015": "-",  # - horizontal bar
    "\u2212": "-",  # - minus sign
    # Ellipsa
    "\u2026": "...",  # ...
    # Nietypowe spacje
    "\u00a0": " ",  # NBSP
    "\u2000": " ",  # en quad
    "\u2001": " ",  # em quad
    "\u2002": " ",  # en space
    "\u2003": " ",  # em space
    "\u2004": " ",  # three-per-em space
    "\u2005": " ",  # four-per-em space
    "\u2006": " ",  # six-per-em space
    "\u2007": " ",  # figure space
    "\u2008": " ",  # punctuation space
    "\u2009": " ",  # thin space
    "\u200a": " ",  # hair space
    "\u202f": " ",  # narrow no-break space
    "\u205f": " ",  # medium mathematical space
    "\u3000": " ",  # ideographic space
    # Low-9 quotes
    "\u201a": ",",  # ,
}

SMART_PATTERN = re.compile("|".join(re.escape(k) for k in REPLACEMENTS))


def should_skip_dir(d: str) -> bool:
    name = os.path.basename(d).lower()
    return name in {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".venv",
        "venv",
        "env",
    }


def list_files(root: Path, exts) -> list[Path]:
    out = []
    for dp, dn, fn in os.walk(root):
        if should_skip_dir(dp):
            continue
        for f in fn:
            p = Path(dp) / f
            if p.suffix.lower() in exts:
                out.append(p)
    return out


def replace_smart(text: str, per_char_counter: dict[str, int]) -> str:
    def _sub(m):
        ch = m.group(0)
        per_char_counter[ch] = per_char_counter.get(ch, 0) + 1
        return REPLACEMENTS[ch]

    return SMART_PATTERN.sub(_sub, text)


def process_file(p: Path, write: bool, backup_dir: Path | None, summary) -> None:
    rel = p.resolve().relative_to(Path.cwd().resolve())
    try:
        raw = p.read_bytes()
    except Exception as e:
        print(f"[READ-ERR] {rel}: {e}")
        summary["read_errors"] += 1
        return

    had_bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        s = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        print(f"[NON-UTF8] {rel}: {e}")
        summary["non_utf8"] += 1
        return

    per_char_counter = {}
    new_s = replace_smart(s, per_char_counter)

    if per_char_counter:
        summary["files_with_smart"] += 1
        summary["chars_found"] += sum(per_char_counter.values())
        print(
            f"[SMART] {rel} -> {sum(per_char_counter.values())} zamian ({', '.join(sorted(set(per_char_counter.keys())))})"
        )

        if write:
            if backup_dir:
                backup_path = backup_dir / rel
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    backup_path.write_bytes(raw)
                except Exception as e:
                    print(f"[BACKUP-ERR] {rel}: {e}")
                    summary["backup_errors"] += 1
                    return

            enc = "utf-8-sig" if had_bom else "utf-8"
            try:
                p.write_text(new_s, encoding=enc, newline=None)
            except Exception as e:
                print(f"[WRITE-ERR] {rel}: {e}")
                summary["write_errors"] += 1
                return

            summary["files_modified"] += 1
    else:
        summary["clean_files"] += 1


def main():
    ap = argparse.ArgumentParser(
        description="Skan i zamiana smart cudzysłowów/pauz na ASCII w repo."
    )
    ap.add_argument("--root", default=".", help="Katalog startowy (domyślnie: .)")
    ap.add_argument(
        "--write",
        action="store_true",
        help="Wykonaj realne zmiany (bez tej flagi tylko skan).",
    )
    ap.add_argument(
        "--exts",
        default=",".join(sorted(DEFAULT_EXTS)),
        help="Rozszerzenia plików, np.: .py,.txt,.md (domyślne zestaw).",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Nie twórz backupów (domyślnie: tworzy).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    exts = set(x.strip().lower() for x in args.exts.split(",") if x.strip().startswith("."))
    if not exts:
        exts = DEFAULT_EXTS

    backup_dir = None
    if args.write and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = root / f".backup_smart_punct_{stamp}"

    files = list_files(root, exts)

    summary = {
        "total_files": len(files),
        "files_with_smart": 0,
        "files_modified": 0,
        "clean_files": 0,
        "non_utf8": 0,
        "read_errors": 0,
        "write_errors": 0,
        "backup_errors": 0,
        "chars_found": 0,
    }

    print(f"Skanuję {len(files)} plików z rozszerzeniami: {', '.join(sorted(exts))}")
    if args.write:
        if backup_dir:
            print(f"Backup zmienianych plików: {backup_dir}")
        print("TRYB: ZAPIS (zamiana znaków w plikach).")
    else:
        print("TRYB: SKAN (bez zmian w plikach).")

    for p in files:
        process_file(p, args.write, backup_dir, summary)

    print("\nPodsumowanie:")
    print(f"  Plików łącznie:          {summary['total_files']}")
    print(f"  Zawierały smart-znaki:   {summary['files_with_smart']}")
    print(f"  Zmienione (write):       {summary['files_modified']}")
    print(f"  Czyste:                  {summary['clean_files']}")
    print(f"  NON-UTF8:                {summary['non_utf8']}")
    print(f"  Błędy odczytu/zapisu:    {summary['read_errors']}/{summary['write_errors']}")
    print(f"  Błędy backupu:           {summary['backup_errors']}")
    print(f"  Łącznie zamian znaków:   {summary['chars_found']}")


if __name__ == "__main__":
    main()
