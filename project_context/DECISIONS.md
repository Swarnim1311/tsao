# TSAO — Decision Log (binding for future sessions)

**Last updated:** 2026-10-08 10:35 UTC (D20 website demo)

Do not reverse these without new evidence + explicit changelog entry.

## D01 — Official Sen1Floods11 splits are frozen (2026-10-07)
Train 252 / Valid 89 / Test 90 / Bolivia-holdout 15. Verified zero overlap, zero missing/unreferenced (NB01). Never re-split randomly; never move chips between splits.

## D02 — Bolivia is a holdout, never training (2026-10-07)
Bolivia's 15 chips may be preprocessed for evaluation but must never enter training data, normalization statistics, or preprocessing parameter fitting.

## D03 — S1 chip is the canonical target grid (2026-10-07)
Exact S1 shape/transform/CRS/bounds preserved; S1 never reprojected. DEM and IMERG are mapped onto it pixel-for-pixel (512×512).

## D04 — Labels are already aligned; never resample/reproject them (2026-10-07)
Verified CRS/shape/transform/bounds agreement (NB01). `-1` = ignore (valid_mask `label != -1`), excluded from loss. `0/1` preserved exactly; `y` stores 0/1 with invalid set to 0 but masked.

## D05 — DEM: bilinear resample of mosaicked intersecting tiles (2026-10-07)
Why: Copernicus GLO-30 is continuous elevation → bilinear preserves physical values better than nearest; multi-tile overlap requires mosaicking. Physical metres stored; no normalization in NB02. DEM `nodata` attr is None → non-finite = invalid.

## D06 — IMERG: `precipitation` only, antecedent t−3/t−2/t−1, bilinear (2026-10-07)
Why: isolate antecedent-moisture signal; forbid event-day/future leakage. Per-chip event date from STAC catalog `properties.datetime` (authoritative, verified one date per event). Explicit lon/lat coordinate use; orientation detected at runtime (files here are lat-ascending). Bilinear for consistency with DEM. Document 0.1° coarseness caveat (chip ≈ 0.046° < one IMERG cell).

## D07 — S1 NaN handling: mask, then per-chip-mean fill (2026-10-07)
Why: NaN ≠ 0 dB (physical value). `s1_valid = isfinite(VV) & isfinite(VH)` recorded; NaNs filled with per-chip valid-mean (fallback 0.0) so tensors are finite; `valid_mask` excludes them from loss. Environmental non-finite values also folded into `valid_mask`.

## D08 — No normalization in NB02; train-only stats later (2026-10-07)
Why: any global/train+test statistics would leak. NB02 stores physical values. NB03 must compute norm stats from TRAIN split only.

## D09 — One compressed `.npz` per chip + metadata (2026-10-07)
Why: 446 × (6×512×512 float32 ≈ 6 MB) ≈ 2.7 GB uncompressed total — per-chip files keep RAM bounded and allow split-wise loading. Contents: `X` float32, `y` uint8, `valid_mask` bool, scalar metadata. Plus `metadata.csv` + `dataset_summary.json`. Reruns safe via `OVERWRITE_EXISTING=False`.

## D10 — NB01 locked (2026-10-07)
Audit-only notebook, 25/25 PASS. Modifiable only for genuine dependency-breaking issues.

## D11 - Standardization with train-only stats, dynamic at load (2026-10-07 09:15 UTC)
Why: (x-mean)/std per channel from 252 train chips (valid_mask px); y/mask never normalized; .npz stay physical; post-norm sanitize non-finite->0 (=mean); y float32 for BCE/Dice; batch 4 / workers 0 / pin iff CUDA.

## D12 - NB04 image-only baseline (2026-10-07 09:52 UTC)
S1-only UNet-32, masked BCE+Dice, val-IoU selection, test-once; baseline for NB05/NB06, no superiority claims.

## D13 - NB05 early-fusion M3 comparison (2026-10-07 10:32 UTC)
Controlled NB04 replica, only first conv 6ch; M3 established for NB06 ablations; verdict from evidence only.

## D14 - NB06 ablation verdict (2026-10-08 05:22 UTC)
Shared-protocol M0/M1/M2/M3; M0/M3 locked-not-retrained; single-seed suggestive language only.

## D14 amendment - overfit-gate calibration (2026-10-08 05:23 UTC)
§10 1-batch overfit gate widened 60 -> 150 steps (threshold <0.60 unchanged) after measured evidence (M1 crosses at step 132, M2 at 91; user-approved). Training protocol itself (seed/loss/opt/sched/patience/thr/AMP/splits) unchanged; not a retune of M1/M2.

## D15 - NB07 Bolivia inference-only verdict (2026-10-08 05:38 UTC)
Locked M0/M1/M2/M3 evaluated on 15-chip holdout; no training/tuning; n=15 suggestive only.

## D16 - NB08 threshold/distribution-shift framing (2026-10-08 05:47 UTC)
Valid-only threshold selection; Bolivia/test never select thresholds. ±0.005 band = single-run variation, never a geographic-shift yardstick. M1 -0.0388 Bolivia drop = moderate geographic-generalization drop with best holdout rank.

## D17 - NB09 multiseed + calibration framing (2026-10-08 07:48 UTC)
Seed 42 locked-reuse; seeds 7/123 identical-protocol retrains. Conclusions require seed consistency + test + Bolivia + gap + morphology + calibration. No significance claims with 3 seeds / n=15; no architectural claims yet.

## D18 - NB10 final-model decision (2026-10-08 08:24 UTC)
ONE architecture experiment (Attention U-Net, M1 inputs/protocol). FINAL MODEL = M1 by pre-registered rule (test>=M1+0.005 AND Bolivia>=M1-0.01). No further architecture experiments.

## D19 - M1 locked production model + reusable inference pipeline (2026-10-08 09:43 UTC)
M1 (UNet-32, VV/VH/DEM, models/nb06_m1_s1_dem_best.pt, thr 0.5, train-only stats) is the production model. Inference lives in src/inference/ (FloodPredictor + CLI + GeoTIFF + latitude-corrected WGS84 area). Pipeline reproduces locked test metrics (diff ~1e-6); area error is descriptive only, never a tuning signal. No retraining/retuning/new architectures from this point without a new decision ID.

## D20 - T-SAO website demo rules (2026-10-08 10:35 UTC)
Website (website/backend + website/frontend) shows only real inference and real locked values: no fake feeds/scores/maps, no invented scene metadata, no model logic in the frontend, benchmark metrics always labelled as ground-truth evaluation. 4 demo scenes are fixed test chips; locked research files stay read-only.

## D21 - T-SAO visual redesign rules (2026-10-08 11:30 UTC)
Frontend-only recomposition (website/frontend/{index.html,styles.css,app.js}); backend app.py/layers.py, src/inference/, models, processed, notebooks all frozen. No fake data/feeds; palette locked to warm-black/ivory/single-teal + sand; neumorphism/neon/glassmorphism explicitly rejected. API/SSE contracts immutable.

## D22 - T-SAO command-center art direction (2026-10-08 12:10 UTC, supersedes D21 palette/type)
Command aesthetic: Space Grotesk + Inter + JetBrains Mono; graphite/navy + aqua/lime/amber semantic palette; img1 wave as fixed environment (frontend/bg.jpg, optimized); glass only for nav/deck/status/pipeline; API/SSE contracts and all locked files still immutable. Screenshots in website/tsao_*.png.
