#!/usr/bin/env python3
"""Generate final Figure 3 panels (A-C).

Inputs:
- fig1_scanvi_analyzed.h5ad (contains X_umap, scanvi_pred, and relevant per-cell metadata columns)

Outputs:
- Figure 3A: 4 score UMAPs (MIX only)
- Figure 3B: scatter + regression per identity (MIX only)
- Figure 3C: KDE contour density (MIX only)
"""

import argparse
import os

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl

import statsmodels.api as sm
from scipy.stats import pearsonr, gaussian_kde


def ensure_umap(a: sc.AnnData):
    if "X_umap" not in a.obsm:
        if "X_scanvi" in a.obsm:
            sc.pp.neighbors(a, use_rep="X_scanvi")
            sc.tl.umap(a)
        else:
            raise ValueError("X_umap not found and X_scanvi not present. Provide an h5ad with UMAP or scANVI latent.")


def ensure_module_scores(adata: sc.AnnData):
    """Compute required module scores for Figure 3 if missing.

    This reproduces the manuscript notebook logic (review_figure_v2.ipynb, cells 24–25).
    """
    # 1. Define Gene Modules (Signatures)
    # These markers are standard for the cell lines and are expected to be in the panel.
    signatures = {
        "Androgen_Response": ["KLK3", "AR", "FKBP5",  "TMPRSS2", "NKX3-1"],  # LNCaP
        "Estrogen_Response": ["ESR1", "PGR", "GATA3", "FOXA1", "TFF1", "GREB1"],    # MCF7
        "HER2_Signaling":    ["ERBB2", "GRB7", "STARD3", "PGAP3", "CDH1"],           # SKBR3
        "Cell_Cycle_G2M":    ["MKI67", "TOP2A", "CDK1", "CCNB1", "CENPF", "AURKA"],  # Proliferation
    }

    available_genes = set(adata.var_names)
    print("Scoring gene modules...")
    for sig_name, genes in signatures.items():
        # If already present, do not recompute
        if sig_name in adata.obs.columns:
            continue
        valid_genes = [g for g in genes if g in available_genes]
        if len(valid_genes) > 2:
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=sig_name)
            print(f"  - {sig_name}: {len(valid_genes)} genes found.")
        else:
            print(f"  - {sig_name}: SKIPPED (only {len(valid_genes)} genes found).")

    # Force HER2 scoring (even if few genes found)
    her2_genes = ["ERBB2", "GRB7", "STARD3", "PGAP3", "CDH1"]
    valid_her2 = [g for g in her2_genes if g in available_genes]
    if len(valid_her2) > 0:
        sc.tl.score_genes(adata, gene_list=valid_her2, score_name="HER2_Signaling")
        print(f"Calculated HER2_Signaling ({len(valid_her2)} genes) [Forced]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="stamp_fig1_scanvi_outputs/fig1_scanvi_analyzed.h5ad", help="Input analyzed h5ad.")
    ap.add_argument("--outdir", default="figures/Figure3", help="Output directory for Figure 3 panels.")
    args = ap.parse_args()

    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    ensure_umap(adata)

    # Ensure Figure 3 module score columns exist
    ensure_module_scores(adata)

    # ----------------------------
    # Figure 3A panel (UMAP scores)
    # ----------------------------
    # (Code kept in manuscript-final format)
    # ----------------------------
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.titlesize"] = 10

    # ----------------------------
    # SETTINGS
    # ----------------------------
    os.makedirs(OUTDIR, exist_ok=True)
    FIG3A_BASENAME = "fig3a_mix_signature_umap_full"

    SAMPLE_KEY = "sample"
    MIX_REGEX = "MIX"
    UMAP_KEY = "X_umap"

    SCORE_TITLES = {
        "Androgen_Response": "Androgen Response\n(LNCaP)",
        "Estrogen_Response": "Estrogen Response\n(MCF7)",
        "HER2_Signaling":    "HER2 Signaling\n(SKBR3)",
        "Cell_Cycle_G2M":    "G2/M Cell Cycle"
    }
    SCORES = list(SCORE_TITLES.keys())

    POINT_SIZE = 3.0
    CMAP = "magma"
    CLIP_Q = (0.01, 0.99) # This clips COLORS, not positions

    # ----------------------------
    # DATA PREP
    # ----------------------------
    mix_mask = adata.obs[SAMPLE_KEY].astype(str).str.contains(MIX_REGEX, case=False, regex=True)
    mix = adata[mix_mask].copy()

    if UMAP_KEY not in mix.obsm:
        raise ValueError("Run UMAP first!")

    coords = mix.obsm[UMAP_KEY]
    x_all = coords[:, 0]
    y_all = coords[:, 1]

    # --- THE FIX: USE MIN/MAX + PADDING ---
    # Instead of percentiles (which trim data), use absolute min/max with a 5% buffer
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    # ----------------------------
    # PLOT
    # ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.0))
    axes = axes.ravel()

    for i, (ax, score) in enumerate(zip(axes, SCORES)):
        if score not in mix.obs.columns:
            print(f"Skipping {score}: not found.")
            continue

        vals = mix.obs[score].values.astype(float)

        # Clip colors (robust contrast)
        vmin = np.nanquantile(vals, CLIP_Q[0])
        vmax = np.nanquantile(vals, CLIP_Q[1])
        vals_clip = np.clip(vals, vmin, vmax)

        # Sort points (Signal on top)
        order = np.argsort(vals_clip)
        x_sorted = x_all[order]
        y_sorted = y_all[order]
        c_sorted = vals_clip[order]

        sca = ax.scatter(
            x_sorted, y_sorted,
            c=c_sorted,
            s=POINT_SIZE,
            linewidths=0,
            edgecolors='none',
            rasterized=True,
            cmap=CMAP,
            vmin=vmin, vmax=vmax
        )

        ax.set_title(SCORE_TITLES[score], fontweight="bold", pad=10)
        ax.set_xlim(xlim) # Applies the padded full range
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.axis('off')

        # Colorbar
        cb = fig.colorbar(sca, ax=ax, fraction=0.03, pad=0.02, aspect=20)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=7, length=2, width=0.5)

        # Simple Axis labels on bottom-left plot only
        if i == 2: 
            ax.text(0.02, 0.02, "UMAP 1", transform=ax.transAxes, fontsize=8, ha='left')
            ax.text(0.02, 0.08, "UMAP 2", transform=ax.transAxes, fontsize=8, ha='left', rotation=90)

    plt.tight_layout()

    png_path = os.path.join(OUTDIR, f"{FIG3A_BASENAME}.png")
    pdf_path = os.path.join(OUTDIR, f"{FIG3A_BASENAME}.pdf")

    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    print(f"Saved to {png_path}")


    # ----------------------------
    # Figure 3B (scatter + regression by identity)
    # ----------------------------
    # ----------------------------
    # USER SETTINGS
    # ----------------------------
    os.makedirs(OUTDIR, exist_ok=True)

    SAMPLE_KEY = "sample"
    MIX_REGEX  = "MIX"          # keep MIX only
    ID_KEY     = "scanvi_pred"  # identity: LnCAP/MCF7/SKBR3
    XCOL       = "Area.um2"     # x-axis
    YCOL       = "Cell_Cycle_G2M"  # y-axis (module score)

    GROUP_ORDER = ["LnCAP", "MCF7", "SKBR3"]

    # marker aesthetics (publishable)
    POINT_SIZE   = 6
    POINT_ALPHA  = 0.8
    EDGE_COLOR   = "black"
    EDGE_WIDTH   = 0.35
    FACE_COLOR   = "black"   # open circles
    LINE_COLOR   = "#2b6cb0"  # nice blue (edit if you want)
    LINE_WIDTH   = 2.2

    # if you have too many points, downsample per group for visual clarity
    MAX_PER_GROUP = 30000   # set None to disable downsampling

    # ----------------------------
    # PREP DATA (MIX only)
    # ----------------------------
    mix_mask = adata.obs[SAMPLE_KEY].astype(str).str.contains(MIX_REGEX, case=False, regex=True)
    mix = adata[mix_mask].copy()

    need = [XCOL, YCOL, ID_KEY]
    missing = [c for c in need if c not in mix.obs.columns]
    if missing:
        raise ValueError(f"Missing columns in mix.obs: {missing}")

    df = mix.obs[need].dropna().copy()
    df[ID_KEY] = df[ID_KEY].astype(str)

    # keep only groups in data (and preserve order)
    groups = [g for g in GROUP_ORDER if g in df[ID_KEY].unique()]

    # optional downsample per identity to keep PDF sizes reasonable + improve readability
    if MAX_PER_GROUP is not None:
        df = df.groupby(ID_KEY, group_keys=False).apply(
            lambda x: x.sample(min(len(x), MAX_PER_GROUP), random_state=0)
        )

    # global limits for consistent axes
    xlim = (df[XCOL].quantile(0.001), df[XCOL].quantile(0.999))
    ylim = (df[YCOL].quantile(0.001), df[YCOL].quantile(0.999))

    # ----------------------------
    # STATS HELPER (slope + 95% CI + r)
    # ----------------------------
    def fit_line_stats(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]; y = y[ok]
        if len(x) < 20:
            return None

        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        slope = float(model.params[1])
        intercept = float(model.params[0])

        # conf_int can be ndarray or DataFrame depending on versions/settings
        ci = model.conf_int()
        if hasattr(ci, "iloc"):
            ci_lo, ci_hi = ci.iloc[1].tolist()
        else:
            ci_lo, ci_hi = ci[1].tolist()

        r, p = pearsonr(x, y)
        r2 = float(model.rsquared)

        return {
            "n": len(x),
            "slope": slope,
            "intercept": intercept,
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "r": float(r),
            "p": float(p),
            "r2": r2
        }

    # ----------------------------
    # PLOT: open-circle scatter per identity
    # ----------------------------
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.8), sharex=True, sharey=True)

    if len(groups) == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        sub = df[df[ID_KEY] == g]

        # open-circle scatter
        ax.scatter(
            sub[XCOL].values, sub[YCOL].values,
            s=POINT_SIZE,
            facecolors=FACE_COLOR,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            alpha=POINT_ALPHA,
            rasterized=True
        )

        st = fit_line_stats(sub[XCOL].values, sub[YCOL].values)
        if st is not None:
            # fit line
            xx = np.linspace(xlim[0], xlim[1], 200)
            yy = st["intercept"] + st["slope"] * xx
            ax.plot(xx, yy, color=LINE_COLOR, linewidth=LINE_WIDTH)

            # annotate (slope per 100 um^2 is easier to read)
            slope_100 = st["slope"] * 100.0
            ci_lo_100 = st["ci_lo"] * 100.0
            ci_hi_100 = st["ci_hi"] * 100.0

            ax.text(
                0.03, 0.97,
                f"r = {st['r']:.2f}\n"
                f"slope = {slope_100:+.2f} / 100µm²\n"
                f"95% CI [{ci_lo_100:+.2f}, {ci_hi_100:+.2f}]",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9
            )

        ax.set_title(g, fontsize=11)
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    axes[0].set_ylabel("G2/M module score", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Cell Area (µm²)", fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # plt.suptitle("Morpho–transcriptomic coupling in MIX", y=1.02, fontsize=12)
    plt.tight_layout()

    out_png = os.path.join(OUTDIR, "fig3_scatter_open_circles_by_identity.png")
    out_pdf = os.path.join(OUTDIR, "fig3_scatter_open_circles_by_identity.pdf")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    print("Saved:")
    print(" -", out_png)
    print(" -", out_pdf)
    print("Rows plotted:", len(df))


    # ----------------------------
    # Figure 3C (KDE density contours)
    # ----------------------------
    # (Code kept in manuscript-final format)
    # ----------------------------
    # -----------------------
    # USER SETTINGS
    # -----------------------
    OUTDIR = OUTDIR  # uses your existing OUTDIR variable
    BASENAME = "fig3b_morpho_density_mix"
    SAMPLE_KEY = "sample"
    MIX_REGEX = "MIX"
    ID_KEY = "scanvi_pred"          # LnCAP / MCF7 / SKBR3
    XCOL = "Area.um2"
    YCOL = "Cell_Cycle_G2M"         # your module score column
    THRESH = 10                     # try 5 or 10
    USE_THRESHOLD = False            # set False to show all cells

    # Color palette (match your UMAP colors if you want)
    PALETTE = {"LnCAP": "#1f77b4", "MCF7": "#ff7f0e", "SKBR3": "#2ca02c"}

    # KDE grid resolution
    GRID_N = 220

    # Contour levels (as fractions of max density per group)
    LEVELS = [0.10, 0.25, 0.50, 0.75]

    # -----------------------
    # PREP DATA (MIX only)
    # -----------------------
    mix_mask = adata.obs[SAMPLE_KEY].astype(str).str.contains(MIX_REGEX, case=False, regex=True)
    mix = adata[mix_mask].copy()

    need = [XCOL, YCOL, ID_KEY]
    missing = [c for c in need if c not in mix.obs.columns]
    if missing:
        raise ValueError(f"Missing columns in mix.obs: {missing}")

    df = mix.obs[[XCOL, YCOL, ID_KEY]].dropna().copy()
    df[ID_KEY] = df[ID_KEY].astype(str)

    # keep only expected groups (and keep order)
    groups = [g for g in ["LnCAP", "MCF7", "SKBR3"] if g in df[ID_KEY].unique()]
    if len(groups) < 2:
        raise ValueError(f"Need >=2 identities in MIX for comparison. Found: {groups}")

    # optional gating
    if USE_THRESHOLD:
        df = df[df[YCOL] >= THRESH].copy()

    # -----------------------
    # KDE helper
    # -----------------------
    def compute_kde_on_grid(x, y, xgrid, ygrid):
        """Return KDE evaluated on meshgrid."""
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        return X, Y, Z

    # grid bounds (use ALL df for consistent axes)
    x_min, x_max = df[XCOL].min(), df[XCOL].max()
    y_min, y_max = df[YCOL].min(), df[YCOL].max()

    # Add a small padding so contours don't touch edges
    x_pad = 0.03 * (x_max - x_min) if x_max > x_min else 1
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1
    xgrid = np.linspace(x_min - x_pad, x_max + x_pad, GRID_N)
    ygrid = np.linspace(y_min - y_pad, y_max + y_pad, GRID_N)

    # -----------------------
    # PLOT
    # -----------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.6))

    # If you want a subtle background of all gated points:
    ax.scatter(df[XCOL].values, df[YCOL].values, s=2, c="#d0d0d0", alpha=0.15, linewidths=0, rasterized=True)

    for g in groups:
        sub = df[df[ID_KEY] == g]
        if sub.shape[0] < 200:
            # too few cells for stable KDE; skip or fall back to scatter
            ax.scatter(sub[XCOL], sub[YCOL], s=4, alpha=0.35, c=PALETTE.get(g, "black"),
                       linewidths=0, rasterized=True, label=f"{g} (n={sub.shape[0]})")
            continue

        X, Y, Z = compute_kde_on_grid(sub[XCOL].values, sub[YCOL].values, xgrid, ygrid)

        # Convert LEVELS into absolute density thresholds per group
        zmax = np.max(Z)
        levels_abs = [zmax * lv for lv in LEVELS]

        # Draw contours (publishable, avoids muddy fills)
        ax.contour(X, Y, Z, levels=levels_abs, colors=[PALETTE.get(g, "black")], linewidths=1.6)
        # Optionally add a light fill only for the highest density region:
        ax.contourf(X, Y, Z, levels=[levels_abs[-1], zmax], colors=[PALETTE.get(g, "black")], alpha=0.10)

    # Labels + title
    # title = "Morpho-transcriptomic coupling in MIX"
    # if USE_THRESHOLD:
    #     title += f" (G2/M ≥ {THRESH})"
    # ax.set_title(title)

    ax.set_xlabel("Cell Area ($\\mu m^2$)", fontsize=16)
    ax.set_ylabel("G2/M module score", fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=14)

    # Legend (manual, so it's always clean)
    handles = []
    labels = []
    for g in groups:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0],[0], color=PALETTE.get(g,"black"), lw=2))
        labels.append(g)
    ax.legend(handles, labels, frameon=False, title="Identity", 
              loc="upper left", fontsize=12, title_fontsize=13)

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    png_path = os.path.join(OUTDIR, f"{BASENAME}.png")
    pdf_path = os.path.join(OUTDIR, f"{BASENAME}.pdf")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    print("Saved:")
    print(" -", png_path)
    print(" -", pdf_path)
    print("Rows plotted:", df.shape[0], "| Threshold:", THRESH if USE_THRESHOLD else "None")


if __name__ == "__main__":
    main()
