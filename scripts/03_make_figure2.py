#!/usr/bin/env python3
"""Generate final Figure 2 panels (A-E).

Inputs:
- fig1_scanvi_analyzed.h5ad (contains X_umap, scanvi_pred, scanvi_maxprob, and counts in layers['counts'])

Outputs:
- Panel_A / Panel_B / Panel_C / Panel_D / Panel_E_Compact (PDF + PNG)
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


# ----------------------------
# DEFAULTS (match manuscript)
# ----------------------------
SAMPLE_KEY = "sample"
LABEL_KEY  = "scanvi_pred"
CONF_KEY   = "scanvi_maxprob"

PURE_SAMPLES = ["STAMP_02_LNCaP", "STAMP_03_MCF7", "STAMP_04_SKBR3"]
PRED_GROUPS  = ["LnCAP", "MCF7", "SKBR3"]

PALETTE_LABEL = {"LnCAP": "#1f77b4", "MCF7": "#ff7f0e", "SKBR3": "#2ca02c"}
PALETTE_PURE  = {"STAMP_02_LNCaP": "#1f77b4", "STAMP_03_MCF7": "#ff7f0e", "STAMP_04_SKBR3": "#2ca02c"}

FOV_CANDIDATES = ["fov","FOV","fov_id","FOV_id","fovID","Fov","fov_name","fovName"]

TOP_N_PER_GROUP = 8
MAX_TOTAL_GENES = 24


# ----------------------------
# Helpers (for Panel E)
# ----------------------------
def _norm_name(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode("utf-8", errors="ignore")
    else:
        x = str(x)
    if x.startswith("b'") and x.endswith("'"):
        x = x[2:-1]
    return x.strip()

def top_genes_from_rgg(mix_de: sc.AnnData, group_order, rg_key: str,
                       top_n_per_group: int, max_total: int):
    rg = mix_de.uns[rg_key]
    groups = list(rg["names"].dtype.names)

    var_map = {_norm_name(v): v for v in mix_de.var_names}
    genes_union = []

    for g in group_order:
        if g not in groups:
            continue
        count = 0
        for nm in rg["names"][g]:
            nm2 = _norm_name(nm)
            if nm2 in var_map:
                gene = var_map[nm2]
                if gene not in genes_union:
                    genes_union.append(gene)
                    count += 1
            if count >= top_n_per_group:
                break

    genes_union = genes_union[:max_total]
    if len(genes_union) == 0:
        exg = group_order[0]
        raise ValueError(
            "Could not match any DE gene names to var_names.\n"
            f"Example rg names ({exg}): {list(rg['names'][exg][:5])}\n"
            f"Example var_names: {list(mix_de.var_names[:5])}"
        )
    return genes_union

def compute_dotplot_stats(mix_de: sc.AnnData, genes, group_key: str, group_order):
    X = mix_de[:, genes].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    groups = mix_de.obs[group_key].astype(str).values
    df = pd.DataFrame(X, columns=genes)
    df[group_key] = groups

    mean = df.groupby(group_key)[genes].mean().reindex(group_order)                # groups x genes
    pct  = (df[genes] > 0).groupby(df[group_key]).mean().reindex(group_order)     # groups x genes
    return mean, pct


def ensure_umap(a: sc.AnnData):
    if "X_umap" not in a.obsm:
        if "X_scanvi" in a.obsm:
            sc.pp.neighbors(a, use_rep="X_scanvi")
            sc.tl.umap(a)
        else:
            raise ValueError("X_umap not found and X_scanvi not present. Provide an h5ad with UMAP or scANVI latent.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="stamp_fig1_scanvi_outputs/fig1_scanvi_analyzed.h5ad", help="Input analyzed h5ad.")
    ap.add_argument("--outdir", default="stamp_fig1_scanvi_outputs", help="Output directory (a panels/ subfolder will be created).")
    args = ap.parse_args()

    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)
    PANEL_DIR = os.path.join(OUTDIR, 'panels')
    os.makedirs(PANEL_DIR, exist_ok=True)

    # ----------------------------
    # 1. PUBLICATION STYLE
    # ----------------------------
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype']  = 42
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.titlesize'] = 9
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams['ytick.major.width'] = 0.8

    # deterministic shuffles used in Panel A/B
    np.random.seed(0)

    adata = sc.read_h5ad(args.h5ad)
    ensure_umap(adata)

    # if FOV exists with a different name, alias it to 'fov' for Panel D
    if "fov" not in adata.obs.columns:
        for c in FOV_CANDIDATES:
            if c in adata.obs.columns:
                adata.obs["fov"] = adata.obs[c].astype(str).values
                break

    mix_mask = adata.obs[SAMPLE_KEY].astype(str).str.contains("MIX", case=False)
    pure_mask = ~mix_mask

    coords_pure = adata[pure_mask].obsm["X_umap"]
    coords_mix  = adata[mix_mask].obsm["X_umap"]

    pure_samples = adata[pure_mask].obs[SAMPLE_KEY].astype(str)
    pure_colors = pure_samples.map(PALETTE_PURE).fillna("grey").values

    mix_preds = adata[mix_mask].obs[LABEL_KEY].astype(str)
    mix_colors = mix_preds.map(PALETTE_LABEL).fillna("grey").values
    mix_conf    = adata[mix_mask].obs[CONF_KEY].values.astype(float)

    # ============================================================
    # PANEL A: Pure Reference Samples
    # ============================================================
    figA, ax = plt.subplots(figsize=(3.5, 3.0))

    # Plot
    # Note: No sorting needed for categorical data usually, but randomizing can help avoid stacking
    idx = np.arange(len(coords_pure))
    np.random.shuffle(idx)
    ax.scatter(
        coords_pure[idx, 0], coords_pure[idx, 1], 
        c=pure_colors[idx], 
        s=2.0, linewidths=0, rasterized=True
    )

    # Style
    ax.set_title("Reference cancer cell lines", fontweight="bold", loc="center")
    ax.axis("off") # Removes box and ticks completely
    ax.set_aspect('equal')

    # Clean Legend
    handles = [Patch(facecolor=PALETTE_PURE[k], edgecolor='none', label=k.replace("STAMP_", "")) for k in PURE_SAMPLES]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), 
              frameon=False, title="Sample ID", title_fontsize=8)

    plt.savefig(os.path.join(PANEL_DIR, "Panel_A.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(PANEL_DIR, "Panel_A.png"), dpi=600, bbox_inches="tight")
    plt.show()
    print("Saved Panel A")


    # ============================================================
    # PANEL B: Projected Mixture Labels
    # ============================================================
    figB, ax = plt.subplots(figsize=(3.5, 3.0))

    # 1. Background (Pure samples in light grey to show manifold structure)
    ax.scatter(coords_pure[:, 0], coords_pure[:, 1], c="#e0e0e0", s=1.5, linewidths=0, rasterized=True)

    # 2. Foreground (Mix samples) - Shuffled to prevent color masking
    idx_mix = np.arange(len(coords_mix))
    np.random.shuffle(idx_mix)
    ax.scatter(
        coords_mix[idx_mix, 0], coords_mix[idx_mix, 1], 
        c=mix_colors[idx_mix], 
        s=2.5, linewidths=0, rasterized=True
    )

    # Style
    ax.set_title("Projected Mixture Labels", fontweight="bold", loc="center")
    ax.axis("off")
    ax.set_aspect('equal')

    # Legend
    handles = [Patch(facecolor=PALETTE_LABEL[k], edgecolor='none', label=k) for k in PRED_GROUPS]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), 
              frameon=False, title="Prediction")

    plt.savefig(os.path.join(PANEL_DIR, "Panel_B.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(PANEL_DIR, "Panel_B.png"), dpi=600, bbox_inches="tight")
    plt.show()
    print("Saved Panel B")


    # ============================================================
    # PANEL C: Mapping Confidence (SORTING IS CRITICAL)
    # ============================================================
    figC, ax = plt.subplots(figsize=(3.8, 3.0)) # Slightly wider for colorbar

    # Sort by confidence so high-conf points (yellow) sit on top of low-conf (purple)
    order = np.argsort(mix_conf)
    scC = ax.scatter(
        coords_mix[order, 0], coords_mix[order, 1], 
        c=mix_conf[order], 
        s=2.5, linewidths=0, rasterized=True,
        cmap="viridis", vmin=0.5, vmax=1.0 # Set explicit range for better contrast
    )

    # Style
    ax.set_title("Mapping Confidence", fontweight="bold", loc="center")
    ax.axis("off")
    ax.set_aspect('equal')

    # Professional Colorbar
    cbar = figC.colorbar(scC, ax=ax, fraction=0.04, pad=0.02, aspect=20)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Max Posterior Prob.", rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=7, length=2)

    plt.savefig(os.path.join(PANEL_DIR, "Panel_C.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(PANEL_DIR, "Panel_C.png"), dpi=600, bbox_inches="tight")
    plt.show()
    print("Saved Panel C")


    # ============================================================
    # PANEL D: Composition Boxplot
    # ============================================================
    # Re-extracting data for safety (assumes 'fov' and 'scanvi_pred' cols exist)
    mix_obs = adata[mix_mask].obs
    if "fov" in mix_obs.columns:
        # Calculate fractions per FOV
        counts = mix_obs.groupby(["fov", "scanvi_pred"]).size().unstack(fill_value=0)
        freqs = counts.div(counts.sum(axis=1), axis=0)

        # Melt for plotting
        df_melt = freqs.reset_index().melt(id_vars="fov", var_name="Identity", value_name="Fraction")
        df_melt = df_melt[df_melt["Identity"].isin(PRED_GROUPS)] # Filter if needed

        figD, ax = plt.subplots(figsize=(3.0, 3.0))

        # Boxplot
        sns.boxplot(
            data=df_melt, x="Identity", y="Fraction", order=PRED_GROUPS,
            palette=PALETTE_LABEL, ax=ax, fliersize=0, linewidth=1.0, width=0.6
        )
        # Strip plot (jittered points)
        sns.stripplot(
            data=df_melt, x="Identity", y="Fraction", order=PRED_GROUPS,
            color="black", size=2.5, alpha=0.6, ax=ax, jitter=True
        )

        # Style
        ax.set_title("Composition Stability", fontweight="bold", loc="center")
        ax.set_xlabel("")
        ax.set_ylabel("Fraction per FOV")
        ax.set_ylim(0, 0.6) # Adjust based on data

        # Despine
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.savefig(os.path.join(PANEL_DIR, "Panel_D.pdf"), bbox_inches="tight")
        plt.savefig(os.path.join(PANEL_DIR, "Panel_D.png"), dpi=600, bbox_inches="tight")
        plt.show()
        print("Saved Panel D")
    else:
        print("Skipped Panel D (No 'fov' column)")


    # ============================================================
    # PANEL E: Dotplot of discriminative markers
    # ============================================================

    # Prepare data (MIX only) for DE
    mix = adata[mix_mask].copy()
    mix_de = mix.copy()

    if "counts" in mix_de.layers:
        mix_de.X = mix_de.layers["counts"].copy()
    else:
        print("WARNING: mix_de.layers['counts'] not found; using mix.X as-is for DE.")

    sc.pp.normalize_total(mix_de, target_sum=1e4)
    sc.pp.log1p(mix_de)

    RG_KEY = "rgg_mix_scanvi_final"
    sc.tl.rank_genes_groups(mix_de, groupby=LABEL_KEY, method="wilcoxon", key_added=RG_KEY)

    marker_genes = top_genes_from_rgg(
        mix_de, PRED_GROUPS, RG_KEY,
        top_n_per_group=TOP_N_PER_GROUP,
        max_total=MAX_TOTAL_GENES,
    )

    mean_expr, pct_expr = compute_dotplot_stats(mix_de, marker_genes, LABEL_KEY, PRED_GROUPS)

    # Gene-wise z-score for color (improves contrast)
    mean_mat = mean_expr.T  # genes x groups
    mean_z = (mean_mat - mean_mat.mean(axis=1).values[:, None]) / (mean_mat.std(axis=1).replace(0, np.nan).values[:, None])
    mean_z = mean_z.fillna(0)


    def plot_publishable_dotplot(pct_df, z_df, out_path=None):
        # Align Data
        if pct_df.shape[0] != z_df.shape[1]: pct_df = pct_df.T
        genes = z_df.index
        groups = z_df.columns

        # ---------------------------------------------------------
        # 1. CONTROL GAP WIDTH HERE
        # ---------------------------------------------------------
        # Calculate exact figure size based on data shape
        # Width: 0.5 inches per group (tight) + 1.5 inches for Y-labels
        # Height: 0.25 inches per gene
        fig_width = (len(groups) * 0.5) + 1.5
        fig_height = (len(genes) * 0.2) + 0.5

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # ---------------------------------------------------------
        # 2. PLOT
        # ---------------------------------------------------------
        x_coords, y_coords = np.meshgrid(np.arange(len(groups)), np.arange(len(genes)))
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()

        pct_vals = pct_df.loc[groups, genes].T.values.flatten()
        z_vals = z_df.loc[genes, groups].values.flatten()

        sc = ax.scatter(
            x_flat, y_flat,
            s=pct_vals * 250, 
            c=z_vals,
            cmap="RdBu_r", 
            vmin=-2, vmax=2, 
            edgecolors="none",
            rasterized=True
        )

        # ---------------------------------------------------------
        # 3. STYLING
        # ---------------------------------------------------------
        # X-Axis (Top)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=0, fontweight="bold", fontsize=9)
        ax.xaxis.tick_top()

        # Y-Axis (Left)
        ax.set_yticks(np.arange(len(genes)))
        ax.set_ylim(len(genes)-0.5, -0.5)
        ax.set_yticklabels(genes, fontstyle="italic", fontsize=8)

        # Grid & Spines
        ax.spines[:].set_visible(False)
        # Important: Set x-limits tightly around the data points
        ax.set_xlim(-0.5, len(groups)-0.5) 

        ax.grid(True, which="major", axis="both", linestyle="-", linewidth=0.5, color="#e0e0e0", zorder=0)
        ax.set_axisbelow(True)

        # Legends (Same as before)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.05, aspect=30)
        cbar.outline.set_visible(False)
        cbar.set_label("Mean Exp. (Z-Score)", size=8)
        cbar.ax.tick_params(labelsize=7)

        sizes = [0.25, 0.50, 0.75, 1.0]
        handles = [plt.scatter([],[], s=s*250, c='gray', label=f"{int(s*100)}%") for s in sizes]
        ax.legend(handles=handles, title="% Detected", bbox_to_anchor=(0.5, -0.02), 
                  loc="upper center", ncol=4, frameon=False, fontsize=8, title_fontsize=8)

        # plt.title("E  Panel Discriminative Markers", loc="left", fontweight="bold", y=1.06, fontsize=10)
        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=600, bbox_inches="tight")
            print(f"Saved to {out_path}")
        plt.show()

    # Run
    plot_publishable_dotplot(pct_expr, mean_z, os.path.join(PANEL_DIR,"Panel_E_Compact.png"))
    plot_publishable_dotplot(pct_expr, mean_z, os.path.join(PANEL_DIR,"Panel_E_Compact.pdf"))


if __name__ == "__main__":
    main()
