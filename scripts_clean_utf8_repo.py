#!/usr/bin/env python3
# scripts/clean_utf8_repo.py
import argparse
import os
import sys

SURRO_MIN = 0xD800
SURRO_MAX = 0xDFFF
UTF8_BOM = b"\xef\xbb\xbf"

EXCLUDE_DIRS_DEFAULT = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


def has_byte_ed(data: bytes) -> bool:
    return b"\xed" in data


def decode_best(data: bytes):
    try:
        return data.decode("utf-8"), True, False  # text, ok_utf8, used_surrogatepass
    except UnicodeDecodeError:
        # fallback with surrogatepass
        return data.decode("utf-8", "surrogatepass"), False, True


def contains_surrogates(text: str) -> bool:
    return any(SURRO_MIN <= ord(ch) <= SURRO_MAX for ch in text)


def fix_surrogates(text: str) -> str:
    # Convert surrogate code units to real Unicode scalars
    try:
        return text.encode("utf-16", "surrogatepass").decode("utf-16")
    except Exception:
        return text  # best effort


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


def should_include(path: str, include_exts: set[str]) -> bool:
    lower = path.lower()
    return any(lower.endswith(ext) for ext in include_exts)


def walk_files(root: str, include_exts: set[str], exclude_dirs: set[str]):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fname in filenames:
            path = os.path.join(dirpath, fname)
            if should_include(path, include_exts):
                yield path


def process_file(
    path: str, in_place: bool, make_backup: bool, norm_newlines: bool, strip_bom: bool
):
    try:
        with open(path, "rb") as f:
            original = f.read()
    except Exception as e:
        return {"path": path, "error": f"read-error: {e}"}

    before_had_ed = has_byte_ed(original)

    data = original
    if strip_bom and data.startswith(UTF8_BOM):
        data = data[len(UTF8_BOM) :]

    text, ok_utf8, used_surrogatepass = decode_best(data)
    had_surrogates = contains_surrogates(text)
    if had_surrogates:
        text = fix_surrogates(text)

    if norm_newlines:
        text = normalize_newlines(text)

    new_data = text.encode("utf-8")
    after_had_ed = has_byte_ed(new_data)
    changed = new_data != original

    if changed and in_place:
        if make_backup:
            try:
                with open(path + ".bak", "wb") as b:
                    b.write(original)
            except Exception as e:
                return {"path": path, "error": f"backup-error: {e}"}
        try:
            with open(path, "wb") as w:
                w.write(new_data)
        except Exception as e:
            return {"path": path, "error": f"write-error: {e}"}

    return {
        "path": path,
        "changed": changed,
        "ok_utf8_initially": ok_utf8,
        "used_surrogatepass": used_surrogatepass,
        "had_surrogates": had_surrogates,
        "byte_ed_before": before_had_ed,
        "byte_ed_after": after_had_ed,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Bulk normalize text files to clean UTF-8 and fix surrogate pairs."
    )
    ap.add_argument("--root", default=".", help="Root directory to scan")
    ap.add_argument(
        "--include-ext",
        default=".py",
        help="Comma-separated list of extensions (e.g. .py,.txt,.md)",
    )
    ap.add_argument(
        "--exclude-dirs", default="", help="Comma-separated extra dirs to exclude"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Do not modify files, only report"
    )
    ap.add_argument(
        "--in-place", action="store_true", help="Write changes back to files"
    )
    ap.add_argument(
        "--backup", action="store_true", help="Write .bak backup before overwrite"
    )
    ap.add_argument(
        "--no-newlines-normalize", action="store_true", help="Do not normalize newlines"
    )
    ap.add_argument(
        "--keep-bom", action="store_true", help="Do not strip UTF-8 BOM if present"
    )
    args = ap.parse_args()

    if not args.dry_run and not args.in_place:
        print(
            "Tip: use --dry-run first, then --in-place --backup to apply.",
            file=sys.stderr,
        )

    include_exts = {e.strip() for e in args.include_ext.split(",") if e.strip()}
    exclude_dirs = set(EXCLUDE_DIRS_DEFAULT)
    if args.exclude_dirs:
        exclude_dirs |= {d.strip() for d in args.exclude_dirs.split(",") if d.strip()}

    total = scanned = changed = fixed_ed = 0
    for path in walk_files(args.root, include_exts, exclude_dirs):
        scanned += 1
        res = process_file(
            path,
            in_place=args.in_place,
            make_backup=args.backup,
            norm_newlines=(not args.no_newlines_normalize),
            strip_bom=(not args.keep_bom),
        )
        if "error" in res:
            print(f"[ERROR] {path}: {res['error']}")
            continue

        total += 1
        if res["changed"]:
            changed += 1
        if res["byte_ed_before"] and not res["byte_ed_after"]:
            fixed_ed += 1

        status = []
        if res["changed"]:
            status.append("CHANGED")
        if res["had_surrogates"]:
            status.append("fixed-surrogates")
        if res["byte_ed_before"] and not res["byte_ed_after"]:
            status.append("removed-0xED")
        if res["byte_ed_after"]:
            status.append("0xED-still-present")
        if not status:
            status.append("OK")

        print(f"{path}: {', '.join(status)}")

    print(
        f"\nScanned: {scanned} | Considered: {total} | Changed: {changed} | Files with 0xED fixed: {fixed_ed}"
    )


if __name__ == "__main__":
    main()
