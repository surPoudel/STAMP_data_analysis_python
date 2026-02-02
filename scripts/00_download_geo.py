#!/usr/bin/env python3
"""Download the 4 STAMP Fig1 samples (counts + .qs.gz) from NCBI GEO.

Why this downloader exists
--------------------------
On some HPC / institutional networks, `ftp.ncbi.nlm.nih.gov` can intermittently return
**HTTP 403** when using Python's default urllib user-agent. To make the workflow
reproducible in more environments, this script:

  - sends an explicit browser-like User-Agent header
  - tries the NCBI mirror at `download.ncbi.nlm.nih.gov`
  - supports retries on transient errors

The destination filenames are kept **stable** (the names used throughout this repo),
even if we download from a mirror.
"""

import argparse
import gzip
import os
import sys
import time
import shutil
import urllib.request
import urllib.error


from typing import Optional

SAMPLES = {
    # "GSM8814930": "GSM8814930_Stamp_C_01",
    "GSM8814931": "GSM8814931_Stamp_C_02_LnCAP",
    "GSM8814932": "GSM8814932_Stamp_C_02_MCFF7",
    "GSM8814933": "GSM8814933_Stamp_C_02_MIX",
    "GSM8814934": "GSM8814934_Stamp_C_02_SKBR3",
}

# Requested suffixes.
SUFFIXES = [
    ".qs.gz",
    "_barcodes.tsv.gz",
    "_features.tsv.gz",
    "_matrix.mtx.gz",
]

BASE_URLS = [
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8814nnn/{gsm}/suppl/{filename}",
    "https://download.ncbi.nlm.nih.gov/geo/samples/GSM8814nnn/{gsm}/suppl/{filename}",
    # FTP fallback (try last; can be blocked on some networks)
    "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8814nnn/{gsm}/suppl/{filename}",
]


def _filename_sources(requested: str) -> list[str]:
    """Return candidate SOURCE filenames to try on the server.

    We keep `requested` as the destination name, but may try a few source variants.
    """
    out = [requested]

    # If requested is gz, also try without .gz on the server (rare, but harmless).
    if requested.endswith(".gz"):
        out.append(requested[:-3])

    # Handle the common typo case (MCFF7 vs MCF7) as a *source* fallback.
    if "MCFF7" in requested:
        out.append(requested.replace("MCFF7", "MCF7"))
        if requested.endswith(".gz"):
            out.append(requested.replace("MCFF7", "MCF7")[:-3])

    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def _stream_download(url: str, tmp_path: str, timeout: int, user_agent: str) -> None:
    headers = {"User-Agent": user_agent, "Accept": "*/*"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)  # 1MB chunks


def _gzip_file(src_path: str, dst_path: str) -> None:
    with open(src_path, "rb") as fin, gzip.open(dst_path, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)


def download_file(
    gsm: str,
    requested_filename: str,
    outdir: str,
    overwrite: bool,
    timeout: int,
    max_retries: int,
    user_agent: str,
) -> None:
    """Download one file, trying mirrors and a few source filename variants."""

    dest = os.path.join(outdir, requested_filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if (not overwrite) and os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {os.path.basename(dest)}")
        return

    tmp = dest + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)

    sources = _filename_sources(requested_filename)
    last_error = None  # type: Optional[Exception]

    for src_filename in sources:
        for base_url in BASE_URLS:
            url = base_url.format(gsm=gsm, filename=src_filename)

            for attempt in range(1, max_retries + 1):
                try:
                    print(f"[get ] {url}")
                    _stream_download(url, tmp, timeout=timeout, user_agent=user_agent)

                    # If destination expects gzip but server delivered uncompressed, gzip it locally.
                    if requested_filename.endswith(".gz") and (not src_filename.endswith(".gz")):
                        _gzip_file(tmp, dest)
                        os.remove(tmp)
                        print(f"[info] downloaded source '{src_filename}' and gzipped to '{requested_filename}'")
                    else:
                        os.replace(tmp, dest)
                        if src_filename != requested_filename:
                            print(f"[info] downloaded source '{src_filename}' as '{requested_filename}'")

                    return

                except urllib.error.HTTPError as e:
                    # 404/403 usually won't improve by retrying the exact same URL; try next mirror/variant.
                    last_error = e
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    break  # break retry loop; go to next mirror/variant

                except Exception as e:
                    last_error = e
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    if attempt < max_retries:
                        time.sleep(min(2 ** (attempt - 1), 8))
                        continue
                    break  # go to next mirror/variant

    msg = f"Failed all mirrors/variants for {gsm}: {requested_filename}"
    if last_error is not None:
        msg += f"\n  -> last error: {last_error}"
    raise RuntimeError(msg)


def main():
    ap = argparse.ArgumentParser(
        description="Download the 4 STAMP Fig1 samples (counts + .qs/.qs.gz) from NCBI GEO."
    )
    ap.add_argument("--outdir", default="data/stamp_fig1_samples", help="Output directory.")
    ap.add_argument("--overwrite", action="store_true", help="Re-download even if files exist.")
    ap.add_argument("--timeout", type=int, default=120, help="Per-file download timeout (seconds).")
    ap.add_argument("--max_retries", type=int, default=4, help="Max retries per URL for transient errors.")
    ap.add_argument(
        "--user_agent",
        default="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        help="HTTP User-Agent header to send (helps avoid HTTP 403 in some environments).",
    )
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    failures = 0
    for gsm, base in SAMPLES.items():
        for suffix in SUFFIXES:
            requested_filename = base + suffix
            try:
                download_file(
                    gsm=gsm,
                    requested_filename=requested_filename,
                    outdir=outdir,
                    overwrite=args.overwrite,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    user_agent=args.user_agent,
                )
            except Exception as e:
                failures += 1
                print(f"[FAIL] {requested_filename}\n{e}\n", file=sys.stderr)

    if failures:
        sys.exit(f"{failures} file(s) failed to download.")
    print("\nDone. Files written to:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
