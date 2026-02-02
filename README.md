# STAMP downstream figure reproduction (Figure 2–3)

This repository contains the **minimal, reproducible code** used to generate the **final, publication-ready** panels for **Figure 2 (A–E)** and **Figure 3 (A–C)** from the STAMP dataset.

The plotting code is kept **identical in style/format** to the final manuscript versions; the scripts only add the required data-loading and variable-prep steps.

---

## Repository layout

```
.
├── convert_qs_to_csv.R                 # provided R helper (qs -> metadata.csv)
├── environment.yml                     # conda environment (Python + R)
├── scripts/
│   ├── 00_download_geo.py              # download the 4 STAMP samples from GEO/NCBI FTP
│   ├── 01_prepare_metadata.sh          # gunzip *.qs.gz -> *.qs and run convert_qs_to_csv.R
│   ├── 02_scanvi_reference_mapping.py  # build fig1_scanvi_analyzed.h5ad (scANVI)
│   ├── 03_make_figure2.py              # Figure 2 panels A–E
│   └── 04_make_figure3.py              # Figure 3 panels A–C
└── data/
    └── (empty; populated by download script)
```

---

## 1) Install (conda)

```bash
conda env create -f environment.yml
conda activate stamp-figures
```

> **GPU (optional):** the environment is CPU-first. If you want GPU training for scVI/scANVI, install a matching CUDA-enabled PyTorch build on your system after creating the env.

---

## 2) Download the 4 STAMP samples (counts + metadata)

This downloads the 10x matrices and the `.qs.gz` SingleCellExperiment objects (used to extract per-cell metadata).

```bash
python scripts/00_download_geo.py --outdir data/stamp_fig1_samples
```

Expected files per sample:
- `*_matrix.mtx.gz`
- `*_barcodes.tsv.gz`
- `*_features.tsv.gz`
- `*.qs.gz`

---

## 3) Convert `.qs.gz` → per-cell `*_metadata.csv`

```bash
bash scripts/01_prepare_metadata.sh
```

This will:
1. `gunzip -k` the `.qs.gz` files to `.qs`
2. run `Rscript convert_qs_to_csv.R`
3. write `*_metadata.csv` into `data/stamp_fig1_samples/`

---

## 4) Build the analyzed object (`fig1_scanvi_analyzed.h5ad`)

This step trains **scVI → scANVI** using the three pure samples as references (LnCAP / MCF7 / SKBR3) and treats MIX as **unknown** during training.

```bash
python scripts/02_scanvi_reference_mapping.py   --data_dir data/stamp_fig1_samples   --outdir stamp_fig1_scanvi_outputs
```

Output:
- `stamp_fig1_scanvi_outputs/fig1_scanvi_analyzed.h5ad`

---

## 5) Reproduce the final figures

### Figure 2 (A–E)

```bash
python scripts/03_make_figure2.py   --h5ad stamp_fig1_scanvi_outputs/fig1_scanvi_analyzed.h5ad   --outdir figures/Figure2
```

Outputs (written into an auto-created `panels/` subfolder):
- `figures/Figure2/panels/Panel_A.(pdf|png)`
- `figures/Figure2/panels/Panel_B.(pdf|png)`
- `figures/Figure2/panels/Panel_C.(pdf|png)`
- `figures/Figure2/panels/Panel_D.(pdf|png)` (skipped automatically if no FOV column exists)
- `figures/Figure2/panels/Panel_E_Compact.(pdf|png)`

### Figure 3 (A–C)

```bash
python scripts/04_make_figure3.py   --h5ad stamp_fig1_scanvi_outputs/fig1_scanvi_analyzed.h5ad   --outdir figures/Figure3
```

Note: `scripts/04_make_figure3.py` will compute the required module score columns
(`Androgen_Response`, `Estrogen_Response`, `HER2_Signaling`, `Cell_Cycle_G2M`) on the fly
if they are not already present in the input `.h5ad`.

Outputs:
- `figures/Figure3/fig3a_mix_signature_umap_full.(pdf|png)`
- `figures/Figure3/fig3_scatter_open_circles_by_identity.(pdf|png)`
- `figures/Figure3/fig3b_morpho_density_mix.(pdf|png)`

---

## Notes on reproducibility

- For **pixel-identical** regeneration of figures, you should run with the same package versions and (ideally) the same hardware/software stack.
- The scripts set random seeds where applicable, but **deep-learning training can still introduce small nondeterminism**. If you need exact reproducibility across machines, the recommended approach is to **version and distribute** the final `fig1_scanvi_analyzed.h5ad` (e.g., via a Zenodo record or GitHub Release) and run only the plotting scripts.

---

## Citation

If you use this code, please cite the associated STAMP manuscript for the original dataset accession and mini-review for the python code usage.

---

## Notebooks (ready-to-run)

If you prefer running the workflow interactively, use the notebooks in `notebooks/`:

- `notebooks/Run_all.ipynb` — end-to-end (download → metadata → scANVI → Figure 2 + Figure 3)
- `notebooks/00_download_geo.ipynb`
- `notebooks/01_prepare_metadata.ipynb`
- `notebooks/02_scanvi_reference_mapping.ipynb`
- `notebooks/03_make_figure2.ipynb`
- `notebooks/04_make_figure3.ipynb`

To launch:

```bash
conda activate stamp-figures
jupyter lab
```

Start Jupyter from the **repo root** (recommended) or from the `notebooks/` folder.
