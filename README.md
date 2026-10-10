# T-SAO — Terrain-Sensitive Assessment of Overflow

Pixel-level flood mapping from Sentinel-1 SAR and terrain intelligence.
Final locked model **M1**: U-Net-32, inputs VV/VH/DEM, threshold 0.50.

- Test IoU **0.6699** · F1 **0.8023** · Precision **0.8185** · Recall **0.7868**
- Bolivia geographic holdout IoU **0.6311** · F1 **0.7739**
- Canonical grid 512×512, EPSG:4326 · train-only normalization · latitude-corrected km²

## Run T-SAO Locally

These steps were verified on Windows with Python 3.11 and CPU inference
(no NVIDIA GPU required; a compatible GPU is used automatically if present).

1. Clone the repository and enter it:

```powershell
git clone https://github.com/Swarnim1311/tsao.git
cd tsao
```

2. Create and activate a virtual environment (Python 3.10+; 3.11 verified):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

PyTorch note: the default PyPI `torch` wheel runs CPU inference on any
laptop. For a smaller CPU-only install use
`pip install torch --index-url https://download.pytorch.org/whl/cpu`
*before* `pip install -r requirements.txt`; for NVIDIA GPU support install
the matching CUDA wheel from https://pytorch.org/get-started/locally/
and the pipeline picks CUDA automatically.

4. Verify Python and PyTorch:

```powershell
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

5. Check that the production checkpoint and configuration exist:

```powershell
dir models\nb06_m1_s1_dem_best.pt models\final_model_config.json processed\normalization_stats.json
dir processed\test\Mekong_1443339.npz processed\test\Ghana_1078550.npz
```

6. Run a real inference smoke test (M1, CPU) — expect 18.2236 km²:

```powershell
python scripts/verify_setup.py
python -m src.inference --npz processed/test/Mekong_1443339.npz --output predictions/demo --prefix demo --save-viz
```

7. Start the FastAPI backend (keep this terminal open):

```powershell
cd website/backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

8. Open the website in a browser: <http://127.0.0.1:8000/> — pick a scene
(Mekong_1443339, Ghana_1078550, Spain_7387658, India_44475) and run analysis.
Key routes: `/api/health`, `/api/model-info`, `/api/scenes`,
`POST /api/predict/{scene}` (SSE), `/api/scenes/{scene}/layer/{vv,vh,dem,prob,mask,gt}`,
`/api/scenes/{scene}/export`.

9. Stop the server: press `Ctrl+C` in the backend terminal, then
`deactivate` to leave the virtual environment.

Data note: `processed/normalization_stats.json` plus the 4 demo scene chips
are committed so a clean clone runs end-to-end. The remaining ~442
processed chips are local-only (see `NB02`/`NB03`); `predictions/` and
`website/.cache/` regenerate automatically and are gitignored.

## 1. What T-SAO does

T-SAO maps floods at the pixel level from multimodal remote sensing:
Sentinel-1 SAR backscatter (VV/VH) plus Copernicus DEM elevation, fused
early in a U-Net segmentation model. The binary flood mask is converted to a
geospatial flood extent (km²) using per-pixel WGS84 areas — the number
emergency response actually needs. A live demo website runs the locked model
on real benchmark scenes.

## 2. Problem statement

Floods must be mapped quickly, at fine spatial resolution, from sensors that
see through clouds. SAR provides all-weather backscatter, but water detection
from imagery alone ignores the terrain that governs where water can stand.

## 3. Research gap

> Existing flood-mapping approaches often rely on satellite imagery alone or
> fuse multiple data sources without sufficiently integrating spatial flood
> detection with environmental context and practical flood-area estimation.

T-SAO integrates Sentinel-1 + terrain context + deep learning + geospatial
area estimation: `Sentinel-1 + terrain + deep learning + area estimation`.

## 4. Dataset

Sen1Floods11 v1.1 hand-labeled split (frozen official splits, zero overlap):

| Split | Chips |
|---|---|
| train | 252 |
| valid | 89 |
| test | 90 |
| bolivia (holdout, never trained on) | 15 |
| total | 446 |

- **S1**: 512×512, VV/VH float32 dB, EPSG:4326. NaN = NoData (~2.75%).
- **Labels**: {−1 NoData 13.63%, 0 non-flood 77.22%, 1 flood 9.16%};
  flood = 10.60% of valid pixels; per-chip median only ~2.57% (imbalanced).
- **DEM**: Copernicus GLO-30, bilinear-mosaicked onto the S1 grid (S1 never
  reprojected; it is the canonical target grid).
- **IMERG**: antecedent rainfall t−3/t−2/t−1 investigated as an input
  (see ablations); the final model does **not** use rainfall.
- 11 events incl. Bolivia, Ghana, India, Mekong, Nigeria, Spain.
- 6 chips have zero valid pixels and contribute nothing to loss/metrics.

Full spec: `project_context/DATASET_SPEC.md`. Raw data lives outside this
repo (`C:/FloodProject/...`); `processed/` (446 per-chip `.npz` + train
normalization stats) is gitignored and must be generated with NB02/NB03.

## 5. Input modalities

| ID | Inputs | Channels |
|---|---|---|
| M0 | S1 only | VV, VH |
| M1 (final) | S1 + DEM | VV, VH, DEM |
| M2 | S1 + rain | VV, VH, rain t−3/t−2/t−1 |
| M3 | S1 + DEM + rain | 6ch |

Standardization uses **train-only** statistics (VV μ=−10.449 σ=4.062;
VH μ=−17.316 σ=4.785; DEM μ=151.505 σ=137.359; rains 7.070/9.768,
7.013/8.435, 14.304/23.184), applied dynamically at load; non-finite → 0.

## 6. Model architecture

U-Net, base 32 channels (~7.76M params; M1 exactly 7,763,041), masked
0.5·BCE + 0.5·Dice loss (labels −1 excluded), AdamW, AMP on CUDA, val-IoU
model selection, test evaluated once.

## 7. Training / evaluation methodology

Shared controlled protocol across ablations (seed 42; only the first
convolution differs). Valid-selected operating points; test and Bolivia never
used for selection. Metrics always computed over valid pixels only.
Calibration, small-water morphology, and multiseed (42/7/123) analyses were
run before locking the final model. No significance claims are made from the
3-seed / 15-chip holdout.

## 8. Ablation study (test IoU, shared protocol)

| Inputs | Test IoU | Δ vs S1 |
|---|---|---|
| S1 only (M0) | 0.6684 | — |
| S1 + DEM (**M1 final**) | **0.6699** | +0.0015 |
| S1 + rain (M2) | 0.6390 | −0.0294 |
| S1 + DEM + rain (M3) | 0.6639 | −0.0045 |

Rainfall context hurt (0.1° cells are coarser than a chip); DEM helps slightly
on test and most on the holdout. An Attention U-Net variant (7.89M params)
scored test 0.6606 / Bolivia 0.5004 and did **not** replace M1.

## 9. Final model

**M1** = U-Net-32, VV/VH/DEM, seed 42, threshold 0.50.
Checkpoint `models/nb06_m1_s1_dem_best.pt` + `models/final_model_config.json`
are locked — do not modify, retrain, or retune.

## 10. Final metrics

| Split | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| test (90 chips) | 0.6699 | 0.8023 | 0.8185 | 0.7868 |
| bolivia (15 chips) | 0.6311 | 0.7739 | 0.8899 | 0.6846 |

Pipeline reproduction matches to ~1e-6 (AMP float).

## 11. Bolivia geographic holdout

15 chips, inference-only, never trained on (IoU drops: M0 −0.0517,
M1 −0.0388, M2 −0.0911, M3 −0.3003). M1 ranks best on the holdout, but with
n=15 and one event this is suggestive only — it does not prove global
generalization.

## 12. Flood-area estimation

Per-pixel WGS84 latitude-corrected areas (~97–98 m²/px; never a constant).
Example: Mekong_1443339 GT 18.7210 vs predicted 18.2236 km² (rel. err 2.66%,
descriptive — area error is never a tuning signal).

## 13. Inference pipeline

`src/inference/` (`FloodPredictor`, UNet, CLI, GeoTIFF + visualization +
area): asserts the locked checkpoint (param count), train-split stats,
threshold 0.5, channel order, finite I/O; respects `valid_mask`; sigmoid +
0.5 threshold; latitude-aware area.

```powershell
python -m src.inference --npz processed/test/Mekong_1443339.npz --output predictions/demo --prefix demo --save-viz
```

## 14. Website / demo

FastAPI backend (`website/backend/app.py`, port 8000) calls the real
`FloodPredictor` — no model logic is duplicated in the frontend. Static
dependency-free frontend offers scene selection, 5 visualization modes
(source/probability/flood/truth/compare), zoom/pan, live staged progress
(SSE), count-up km², benchmark evaluation against real labels, and GeoTIFF
export. All imagery and numbers are live inference.

```powershell
cd website/backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/>. Demo scenes (real test chips):
Mekong_1443339 (18.22 km²), Ghana_1078550 (0.01), Spain_7387658 (13.01),
India_44475 (0.00).

## 15. Repository structure

| Path | Contents (tracked) |
|---|---|
| `src/inference/` | Locked production pipeline |
| `models/nb06_m1_s1_dem_best.pt` | Locked M1 checkpoint (~31 MB) |
| `models/*_config.json` | Locked configs (experimental checkpoints excluded via `.gitignore`) |
| `notebooks/NB01–NB11` | Locked research notebooks |
| `results/` | Metrics, histories, figures, inference panels |
| `project_context/` | State, decisions, changelog, dataset spec, next steps |
| `website/` | Backend + frontend + background art |
| `images/` | Design source imagery |

Generated/local-only (gitignored): `processed/`, `predictions/`,
`website/.cache/`, `.opencode/`, `.kilo/`, `__pycache__/`, `.venv/`.

## 16. How to run inference

Requires `processed/` (NB02+NB03 outputs) and Python deps
(torch, numpy, rasterio, affine, matplotlib). See §13.

## 17. How to run the website

Requires the same environment plus fastapi/uvicorn/PIL. See §14.

## 18. Limitations

- Single-seed main results; no significance claims.
- Bolivia: one event, n=15 — suggestive, not global proof.
- Heavy class imbalance (median chip ~2.6% flood); small scattered water is
  under-detected (see NB08/NB09 error analyses).
- IMERG rainfall too coarse to help at chip scale.
- DEM resampling and S1 speckle bound performance.
- Website demo covers 4 fixed test scenes, not global deployment.

## 19. Future work

Multiseed significance testing, additional geographic holdouts, calibration +
morphology for small water, temporal/antecedent-moisture modeling at finer
scale, and operational hardening of the demo service.
