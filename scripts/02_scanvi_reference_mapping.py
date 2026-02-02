#!/usr/bin/env python3
"""STAMP Fig1 downstream analysis (minimal) - build `fig1_scanvi_analyzed.h5ad`

This script:
1) loads 4 STAMP samples (3 pure references + 1 MIX),
2) merges counts with per-cell metadata (from `convert_qs_to_csv.R`),
3) trains scVI -> scANVI (pure samples labeled, MIX = unknown),
4) writes an analyzed AnnData object used for final figure generation.

Outputs:
- <outdir>/fig1_scanvi_analyzed.h5ad
"""

import argparse
import os
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc

import scvi


DEFAULT_SAMPLE_INFO: Dict[str, str] = {
    "STAMP_02_LNCaP": "GSM8814931_Stamp_C_02_LnCAP",
    "STAMP_03_MCF7":  "GSM8814932_Stamp_C_02_MCFF7",
    "STAMP_04_MIX":   "GSM8814933_Stamp_C_02_MIX",
    "STAMP_04_SKBR3": "GSM8814934_Stamp_C_02_SKBR3",
}


def pick_fov_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["fov", "FOV", "fov_id", "FOV_id", "fovID", "Fov", "fov_name", "fovName"]:
        if c in df.columns:
            return c
    return None


def load_one_sample(data_dir: str, prefix: str, sample_id: str) -> ad.AnnData:
    matrix_path   = os.path.join(data_dir, f"{prefix}_matrix.mtx.gz")
    barcodes_path = os.path.join(data_dir, f"{prefix}_barcodes.tsv.gz")
    features_path = os.path.join(data_dir, f"{prefix}_features.tsv.gz")
    metadata_path = os.path.join(data_dir, f"{prefix}_metadata.csv")

    for p in [matrix_path, barcodes_path, features_path, metadata_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    # counts matrix (cells x genes)
    X = sc.read_mtx(matrix_path).X.T.tocsr()

    barcodes = pd.read_csv(barcodes_path, header=None, compression="gzip")[0].astype(str).values
    features = pd.read_csv(features_path, sep="\t", header=None, compression="gzip")[0].astype(str).values
    metadata = pd.read_csv(metadata_path, index_col=0)

    a = ad.AnnData(X=X)
    a.obs_names = barcodes
    a.var_names = features
    a.var_names_make_unique()

    # align metadata to barcodes
    missing = a.obs_names[~pd.Index(a.obs_names).isin(metadata.index)]
    if len(missing) > 0:
        warnings.warn(f"{sample_id}: metadata missing {len(missing)} barcodes; dropping them.")
        keep = pd.Index(a.obs_names).intersection(metadata.index)
        a = a[keep].copy()

    a.obs = metadata.loc[a.obs_names].copy()
    a.obs["sample_id"] = sample_id
    return a


def main():
    ap = argparse.ArgumentParser(description="Build fig1_scanvi_analyzed.h5ad from STAMP Fig1 samples.")
    ap.add_argument("--data_dir", default="data/stamp_fig1_samples", help="Folder containing *_matrix.mtx.gz, *_metadata.csv, etc.")
    ap.add_argument("--outdir", default="stamp_fig1_scanvi_outputs", help="Output directory.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--n_latent", type=int, default=30, help="Latent dimension for scVI/scANVI.")
    ap.add_argument("--scanvi_epochs", type=int, default=30, help="Training epochs for scANVI.")
    ap.add_argument("--min_counts", type=int, default=500, help="Filter: minimum total counts per cell.")
    ap.add_argument("--min_genes", type=int, default=200, help="Filter: minimum detected genes per cell.")
    ap.add_argument("--area_q_lo", type=float, default=0.01, help="Filter: lower quantile for Area.um2 (if present).")
    ap.add_argument("--area_q_hi", type=float, default=0.99, help="Filter: upper quantile for Area.um2 (if present).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # reproducibility knobs
    np.random.seed(args.seed)
    scvi.settings.seed = args.seed
    sc.settings.verbosity = 2

    # -------------------------
    # 1) Load + concat samples
    # -------------------------
    adata_list = []
    for sample_id, prefix in DEFAULT_SAMPLE_INFO.items():
        print(f"Loading {sample_id}  ({prefix})")
        adata_list.append(load_one_sample(args.data_dir, prefix, sample_id))

    adata_all = ad.concat(
        adata_list,
        label="sample",
        keys=list(DEFAULT_SAMPLE_INFO.keys()),
        join="inner",
        index_unique="-",   # ensures unique obs_names
    )

    # preserve raw counts
    adata_all.layers["counts"] = adata_all.X.copy()

    print("Combined:", adata_all)

    # -------------------------
    # 2) QC metrics + filtering
    # -------------------------
    adata_all.obs["n_counts"] = np.asarray(adata_all.X.sum(axis=1)).ravel()
    adata_all.obs["n_genes_by_counts"] = np.asarray((adata_all.X > 0).sum(axis=1)).ravel()

    # Standardize cell area if present (used in Fig.3)
    if "Area.um2" in adata_all.obs.columns:
        adata_all.obs["cell_area"] = adata_all.obs["Area.um2"]
    elif "area" in adata_all.obs.columns:
        adata_all.obs["cell_area"] = adata_all.obs["area"]

    keep = (adata_all.obs["n_counts"] >= args.min_counts) & (adata_all.obs["n_genes_by_counts"] >= args.min_genes)

    if "cell_area" in adata_all.obs.columns:
        lo, hi = adata_all.obs["cell_area"].quantile([args.area_q_lo, args.area_q_hi])
        keep &= (adata_all.obs["cell_area"] >= lo) & (adata_all.obs["cell_area"] <= hi)

    adata = adata_all[keep].copy()
    print("After filtering:", adata)

    # -------------------------
    # 3) scANVI reference mapping
    # -------------------------
    label_map = {
        "STAMP_02_LNCaP": "LnCAP",
        "STAMP_03_MCF7": "MCF7",
        "STAMP_04_SKBR3": "SKBR3",
        "STAMP_04_MIX": "unknown",
    }
    adata.obs["cell_line_label"] = adata.obs["sample"].map(label_map).astype("category")

    # Choose a batch_key that is not the biological label
    batch_key = pick_fov_column(adata.obs)  # use FOV if present
    print("Using batch_key:", batch_key)

    # ensure X is raw counts
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key=batch_key,
        labels_key="cell_line_label",
    )

    scvi_model = scvi.model.SCVI(adata, n_latent=args.n_latent)
    scvi_model.train()

    scanvi_model = scvi.model.SCANVI.from_scvi_model(scvi_model, unlabeled_category="unknown")
    scanvi_model.train(max_epochs=args.scanvi_epochs)

    # Predictions + confidence
    adata.obs["scanvi_pred"] = scanvi_model.predict(adata)
    probs = scanvi_model.predict(adata, soft=True)
    adata.obs["scanvi_maxprob"] = probs.max(axis=1)

    # Latent + UMAP
    adata.obsm["X_scanvi"] = scanvi_model.get_latent_representation(adata)
    sc.pp.neighbors(adata, use_rep="X_scanvi")
    sc.tl.umap(adata)

    out_h5ad = os.path.join(args.outdir, "fig1_scanvi_analyzed.h5ad")
    adata.write_h5ad(out_h5ad)
    print("\nDone. Wrote:", os.path.abspath(out_h5ad))


if __name__ == "__main__":
    main()
