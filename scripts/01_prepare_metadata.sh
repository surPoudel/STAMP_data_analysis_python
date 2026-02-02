#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT}/data/stamp_fig1_samples"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: ${DATA_DIR} not found."
  echo "Run: python scripts/00_download_geo.py --outdir data/stamp_fig1_samples"
  exit 1
fi

# Decompress .qs.gz -> .qs (keep the .gz)
shopt -s nullglob
for f in "${DATA_DIR}"/*.qs.gz; do
  echo "[gunzip] $(basename "$f")"
  gunzip -kf "$f"
done
shopt -u nullglob

# Run the provided R script (writes *_metadata.csv into data/stamp_fig1_samples/)
cd "${ROOT}"
echo "[Rscript] convert_qs_to_csv.R"
Rscript convert_qs_to_csv.R

echo "Done. Metadata CSVs should be in: ${DATA_DIR}"
