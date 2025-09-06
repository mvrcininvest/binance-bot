#!/usr/bin/env python3
import os
from datetime import datetime
from pathlib import Path

EXTS = {
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
SKIP_DIRS = {
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

CANDIDATE_ENCODINGS = ["cp1250", "iso8859_2", "cp1252", "mac_centeuro"]


def should_skip(dp: str) -> bool:
    return os.path.basename(dp).lower() in SKIP_DIRS


def list_files(root: Path):
    for dp, dn, fn in os.walk(root):
        if should_skip(dp):
            continue
        for f in fn:
            p = Path(dp) / f
            if p.suffix.lower() in EXTS:
                yield p


def try_decode(data: bytes):
    # najpierw spróbuj UTF-8 (z BOM)
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    for enc in CANDIDATE_ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return None, None


def main():
    root = Path(".").resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root / f".backup_encoding_{stamp}"
    converted = 0
    failed = 0
    nonutf = 0

    for p in list_files(root):
        b = p.read_bytes()
        try:
            b.decode("utf-8-sig")
            continue  # już UTF-8
        except UnicodeDecodeError:
            nonutf += 1

        text, enc = try_decode(b)
        if text is None:
            print(f"[ENC-FAIL] {p}")
            failed += 1
            continue

        print(f"[CONVERT] {p} : {enc} -> utf-8")
        # backup
        backup_path = backup_root / p.resolve().relative_to(root)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_bytes(b)
        # zapis jako UTF-8 (bez BOM)
        p.write_text(text, encoding="utf-8")
        converted += 1

    print("\nPodsumowanie:")
    print(f"  Nie-UTF8 wykryte: {nonutf}")
    print(f"  Skonwertowane:    {converted}")
    print(f"  Niepowodzenia:    {failed}")
    if converted > 0:
        print(f"  Backup:           {backup_root}")


if __name__ == "__main__":
    main()
