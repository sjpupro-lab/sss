#!/usr/bin/env python3
"""
prepare_wiki_corpus.py — gather wikiextractor output into a single
training file, bounded by a total byte budget.

Example layouts handled:

    C:\\Users\\you\\Downloads\\archive\\fullEnglish\\
        AA\\wiki_00 wiki_01 ... wiki_99
        AB\\wiki_00 ...
        ...

Every file matching `wiki_<digits>` under `--src` is read in
filesystem order and concatenated to the output until `--max-mb`
megabytes have been written. stream_train's `is_skippable` filter
drops the `<doc>` / `</doc>` marker lines automatically, so the raw
wikiextractor output can be fed directly to ai_store_auto without
a separate cleanup pass.

Usage:
  python tools/prepare_wiki_corpus.py \\
      --src  "C:/Users/devil/Downloads/archive/fullEnglish" \\
      --out  data/kaggle_train.txt \\
      --max-mb 50

Typical sizes:
   5 MB  ->  ~2,500 clauses
  10 MB  ->  ~5,000 clauses
  50 MB  ->  ~25,000 clauses
 100 MB  ->  ~50,000 clauses
"""
import argparse
import os
import re
import sys

PAT = re.compile(r"^wiki_\d+$")


def iter_wiki_files(src):
    for root, _, files in os.walk(src):
        for name in sorted(files):
            if PAT.match(name):
                yield os.path.join(root, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="wikiextractor output root")
    ap.add_argument("--out", required=True, help="merged corpus path")
    ap.add_argument("--max-mb", type=float, default=50.0,
                    help="target bytes written, MB (default 50)")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"src not found: {args.src}", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    budget = int(args.max_mb * 1024 * 1024)
    written = 0
    files_used = 0

    with open(args.out, "wb") as out:
        for path in iter_wiki_files(args.src):
            if written >= budget:
                break
            with open(path, "rb") as src:
                data = src.read()
            remaining = budget - written
            if len(data) > remaining:
                data = data[:remaining]
            out.write(data)
            written += len(data)
            files_used += 1

    print(f"wrote {written / 1024 / 1024:.1f} MB ({written:,} bytes)")
    print(f"from {files_used} wiki_* files")
    print(f"to   {args.out}")

    # Quick sanity: how many non-skippable lines did we get?
    non_skip = 0
    total_lines = 0
    with open(args.out, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines += 1
            s = line.strip()
            if not s or s[0] == "<":
                continue
            if len(s) >= 10:
                non_skip += 1
    print(f"     {total_lines:,} total lines, {non_skip:,} usable as clauses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
