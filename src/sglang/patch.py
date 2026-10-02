#!/usr/bin/env python3
"""Patch SGLang scheduler with latency-attack logging instrumentation.

Usage:
    python patch.py [--target-dir <sglang_dir>] [--python <path_to_python>] [--prefix <env_prefix>] [--variant <name>] [--restore]

Options:
    --target-dir <path>  Direct path to the installed sglang package directory.
    --python <path>      Python executable from the environment containing sglang.
    --prefix <path>      Conda/venv prefix containing lib/python*/site-packages/sglang.
    --venv <path>        Backward-compatible alias for --prefix.
    --variant <name>     Patch variant. Use "default" for file_list.csv or "l40s" for file_list_l40s.csv.
    --restore            Restore original files from backup.

Environment:
    TARGET_DIR           Direct path to the installed sglang package directory.
"""

import csv
import glob
import json
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VARIANT = "default"
FILE_LISTS = {
    DEFAULT_VARIANT: "file_list.csv",
    "l40s": "file_list_l40s.csv",
}


def normalize_sglang_path(path):
    if not path:
        return None

    path = os.path.abspath(os.path.expanduser(path))

    direct_candidates = [path]
    if os.path.basename(path) != "sglang":
        direct_candidates.extend(
            [
                os.path.join(path, "sglang"),
                os.path.join(path, "site-packages", "sglang"),
            ]
        )

    for candidate in direct_candidates:
        if os.path.isfile(os.path.join(candidate, "__init__.py")):
            return candidate
        if os.path.isdir(os.path.join(candidate, "srt")):
            return candidate

    return None


def find_sglang_path(target_dir=None, prefix_path=None, python_path=None):
    if target_dir:
        sglang_path = normalize_sglang_path(target_dir)
        if sglang_path:
            return sglang_path
        print(f"ERROR: TARGET_DIR does not point to a valid sglang package: {target_dir}")
        sys.exit(1)

    if python_path:
        code = """
import importlib.util, json, os
spec = importlib.util.find_spec('sglang')
print(json.dumps(os.path.dirname(spec.origin) if spec and spec.origin else None))
"""
        try:
            output = subprocess.check_output(
                [python_path, "-c", code], text=True, stderr=subprocess.DEVNULL
            ).strip()
            sglang_path = normalize_sglang_path(json.loads(output))
            if sglang_path:
                return sglang_path
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
            pass

    if prefix_path:
        for pattern in (
            os.path.join(prefix_path, "lib", "python*", "site-packages", "sglang"),
            os.path.join(prefix_path, "Lib", "site-packages", "sglang"),
        ):
            for sglang_path in glob.glob(pattern):
                sglang_path = normalize_sglang_path(sglang_path)
                if sglang_path:
                    return sglang_path

    candidates = [
        os.getenv("TARGET_DIR"),
        "/root/miniconda3/lib/python3.10/site-packages/sglang",
        "/home/phil/f/sglang/.venv/lib/python3.12/site-packages/sglang",
        "/home/phil/l/latency-attack/.venv/lib/python3.12/site-packages/sglang",
    ]
    for c in candidates:
        sglang_path = normalize_sglang_path(c)
        if sglang_path:
            return sglang_path

    print("ERROR: Cannot find sglang installation. Use TARGET_DIR, --target-dir, --python, or --venv to specify.")
    sys.exit(1)


def get_file_list_path(variant):
    if variant not in FILE_LISTS:
        names = ", ".join(sorted(FILE_LISTS))
        print(f"ERROR: Unknown variant '{variant}'. Available variants: {names}")
        sys.exit(1)

    file_list = os.path.join(SCRIPT_DIR, FILE_LISTS[variant])
    if not os.path.isfile(file_list):
        print(f"ERROR: file list not found for variant '{variant}': {file_list}")
        sys.exit(1)
    return file_list


def read_file_list(variant=DEFAULT_VARIANT):
    entries = []
    with open(get_file_list_path(variant), "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("<-")
            if len(parts) != 2:
                continue
            dest_rel = parts[0].strip()
            src_rel = parts[1].strip()
            entries.append((dest_rel, src_rel))
    return entries


def patch(sglang_path, variant=DEFAULT_VARIANT):
    entries = read_file_list(variant)
    for dest_rel, src_rel in entries:
        src = os.path.join(SCRIPT_DIR, src_rel)
        dest = os.path.join(sglang_path, dest_rel)
        backup = dest + ".bak"

        if not os.path.isfile(src):
            print(f"SKIP: source not found: {src}")
            continue
        if not os.path.isfile(dest):
            print(f"SKIP: destination not found: {dest}")
            continue

        if not os.path.isfile(backup):
            shutil.copy2(dest, backup)
        else:
            print(f"KEEP: existing backup: {backup}")
        shutil.copy2(src, dest)
        print(f"PATCHED: {dest}")
        print(f"  backup: {backup}")


def restore(sglang_path, variant=DEFAULT_VARIANT):
    entries = read_file_list(variant)
    for dest_rel, _ in entries:
        dest = os.path.join(sglang_path, dest_rel)
        backup = dest + ".bak"

        if os.path.isfile(backup):
            shutil.copy2(backup, dest)
            os.remove(backup)
            print(f"RESTORED: {dest}")
        else:
            print(f"SKIP: no backup for {dest}")


if __name__ == "__main__":
    target_dir = os.getenv("TARGET_DIR")
    prefix_path = None
    python_path = None
    variant = os.getenv("SGLANG_PATCH_VARIANT", DEFAULT_VARIANT)
    do_restore = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--target-dir" and i + 1 < len(args):
            target_dir = args[i + 1]
            i += 2
        elif args[i] in ("--prefix", "--venv") and i + 1 < len(args):
            prefix_path = args[i + 1]
            i += 2
        elif args[i] == "--python" and i + 1 < len(args):
            python_path = args[i + 1]
            i += 2
        elif args[i] == "--variant" and i + 1 < len(args):
            variant = args[i + 1]
            i += 2
        elif args[i] == "--restore":
            do_restore = True
            i += 1
        else:
            i += 1

    sglang_path = find_sglang_path(
        target_dir=target_dir, prefix_path=prefix_path, python_path=python_path
    )
    print(f"sglang path: {sglang_path}")
    print(f"patch variant: {variant}")

    if do_restore:
        restore(sglang_path, variant)
    else:
        patch(sglang_path, variant)

    print("\nDone.")
