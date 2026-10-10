# TSAO Flood-Mapping - Project State

**Last updated:** 2026-10-10 (PORTABILITY VERIFIED: clean-clone runtime complete, CPU + website E2E pass; commit pending user action)

## Portability status (2026-10-10)
- Production M1 loads from clean-clone files only (checkpoint + final_model_config.json + committed processed/normalization_stats.json); no retraining, weights untouched.
- Committed runtime data: processed/normalization_stats.json + processed/test/{Mekong_1443339,Ghana_1078550,Spain_7387658,India_44475}.npz (gitignore exceptions; rest of processed/ ignored).
- Repro: requirements.txt (py>=3.10, CPU default / CUDA auto) + scripts/verify_setup.py (all pass) + README Run T-SAO Locally.
- Verified: CPU inference Mekong 18.2236 / Ghana 0.0097 km2; website backend+frontend+SSE+layers+export E2E pass; unknown-scene 404.
- Limitation: GPU inference path not verified on a GPU-less laptop; full 446-chip processed/ not redistributed (needs raw C:/FloodProject data + NB02/NB03).

## Objective
Deep-learning flood mapping from multimodal remote sensing: Sentinel-1 SAR (VV/VH) + Copernicus DEM elevation + NASA IMERG antecedent rainfall → binary flood mask. Sentinel-1 chip is the canonical target grid (512×512, EPSG:4326).

## Source paths (read-only raw data — never modify)
- Sen1Floods11 v1.1: `C:/FloodProject/Sen1Floods11/v1.1`
  - S1: `.../data/flood_events/HandLabeled/S1Hand` (446 × `.tif`)
  - Labels: `.../data/flood_events/HandLabeled/LabelHand` (446 × `.tif`)
  - Splits: `.../splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv` (no header; `S1Hand,LabelHand` per line)
  - Per-chip STAC catalog (authoritative event dates): `.../catalog/sen1floods11_hand_labeled_source/<Event>_<chipid>/<Event>_<chipid>.json` → `properties.datetime`
  - Event metadata: `.../Sen1Floods11_Metadata.geojson` (one `s1_date` per flood event)
- DEM: `C:/FloodProject/DEM` — 65 main tiles `Copernicus_DSM_COG_10_*_DEM.tif` (recursive, one folder per tile)
- IMERG: `C:/FloodProject/IMERG/data` — 36 daily `.nc4` (GPM IMERG Final Run V07B, var `precipitation`, mm/day, 0.1° global grid)
- Workspace: `C:/Users/Swarnim/Desktop/ML projects/tsao` → `notebooks/`, `project_context/`, `processed/`

## Dataset structure (verified, NB01 25/25 PASS)
- 446 hand-labeled S1 chips; splits Train 252 / Valid 89 / Test 90 / Bolivia-holdout 15; zero overlap, zero missing/unreferenced.
- S1: 512×512, 2 bands (VV,VH), float32 dB, EPSG:4326, res ~8.983e-05°, NaN = NoData (~2.75% in 30-chip sample; VV mean −10.44 dB, VH mean −17.59 dB).
- Labels: 512×512, values {−1,0,1}; S1↔label CRS/shape/transform/bounds aligned. Full census: NoData 13.63%, non-flood 77.22%, flood 9.16% of all pixels; flood = 10.60% of valid pixels. Per-chip flood% mean 10.87 / median 2.57 (441 chips with valid px).
- 11 events: Bolivia, Ghana, India, Mekong (=Cambodia), Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA.
- DEM: 65 tiles, all EPSG:4326 float32 3600×3600 (~30 m), all readable; approx elev range −4…1515 m (5-tile sample).
- IMERG: 36 files, unique dates, all contain `precipitation`, dims (time=1, lat=1800, lon=3600); lon ascending, **lat ASCENDING** (−89.95→89.95 — handle orientation explicitly, never assume axis order).


## NB03 result (2026-10-07 09:15 UTC)
- Train-only norm stats (252 chips) + FloodDataset/loaders; all QC + leakage gates PASS.


## NB04 result (2026-10-07 09:52 UTC)
- S1-only UNet baseline: val IoU=0.6325, test IoU=0.6684; 32 epochs.


## NB05 result (2026-10-07 10:32 UTC)
- Multimodal M3: val IoU=0.6288, test IoU=0.6639 (d=-0.0045 vs NB04).


## NB06 result (2026-10-08 05:22 UTC)
- M1 test IoU=0.6699 (d=+0.0015); M2 test IoU=0.6390 (d=-0.0294); M0 0.6684 / M3 0.6639 locked.


## NB07 result (2026-10-08 05:38 UTC)
- Bolivia IoU: M0=0.6167, M1=0.6311, M2=0.5479, M3=0.3636 (15 chips, inference-only, locked ckpts).


## NB08 result (2026-10-08 05:47 UTC)
- Valid-selected thr M0=0.4, M1=0.5, M2=0.5, M3=0.45; M3-Bolivia verdict + distributions in results/nb08_*.


## NB09 result (2026-10-08 07:48 UTC)
- Calibration + small-water morphology + multiseed(42/7/123); verdict in results/nb09_results.json.


## NB10 result (2026-10-08 08:24 UTC)
- Attention U-Net vs M1: test dIoU=-0.0093, Bolivia dIoU=-0.1308. FINAL MODEL = M1.


## FINAL INFERENCE PIPELINE result (2026-10-08 09:43 UTC)
- M1 is the locked production model; reusable pipeline in src/inference/ (FloodPredictor).
- Test reproduction via pipeline: IoU=0.669888 F1=0.802315 (locked 0.6699/0.8023, diff ~1e-6 float).
- Area: latitude-corrected WGS84 per-pixel areas (~97-98 m2/px); e.g. Mekong_1443339 GT 18.7210 vs pred 18.2236 km2 (rel err 2.66%, descriptive).
- Outputs: predictions/{probabilities,masks}/ GeoTIFFs (EPSG:4326), results/final_inference_examples/ panels, CLI `python -m src.inference`, NB11 demo notebook executed clean. 22/22 quality gates PASS.


## T-SAO PREMIUM REFINEMENT result (2026-10-08 13:00 UTC)
- img2 estuary environment (bg2.jpg), full T-SAO name in hero/method/footer/title, premium navbar (entrance, sliding underlines, mobile links), flood verdict + compare shortcut + standby empty state + stagger + sheen, rebuilt footer.
- Validated live, zero console errors: Mekong 18.2236 / Ghana 0.0097; evidence 0.6699/0.8023/0.6311; reduced-motion + tablet + mobile verified; shots website/tsao2_*.png. Backend/inference/locked files untouched.

## D24 - FINAL REPOSITORY AUDIT COMPLETE (2026-10-08 14:15 UTC)
- Verified repo structure for GitHub submission-readiness: .gitignore covers .kilo/, website/.cache/, generated artifacts; tracked tree ~41MB with M1 + configs + NB01–NB11 + results + site. README documents architecture, locked metrics, and startup. No secrets, no experimental checkpoints. Final audit report generated. GitHub commit/push pending user action.

## T-SAO COMMAND REDESIGN result (2026-10-08 12:10 UTC, supersedes 11:30 editorial direction)
- Flood-intelligence command aesthetic: floating pill nav, wave-environment hero (bg.jpg) with live-status glass panel + HUD strip, glass SCENE/MAP/DATA workstation, hover pipeline, animated-bars evidence. Space Grotesk + Inter + JetBrains Mono; aqua/lime/amber semantic palette.
- ui-ux-pro-max: DeFi tri-stack type, HUD thin-lines, selective liquid-glass, subtle reveals, staged loading UX. Parallax/magnetic/tooltips; reduced-motion + reduced-transparency respected.
- Validated live, zero console errors: Mekong 18.2236 / Ghana 0.0097; screenshots website/tsao_{hero,analyze,compare}.png; backend/inference/locked files untouched.

## T-SAO VISUAL REDESIGN result (2026-10-08 11:30 UTC)
- Cinematic full-viewport SAR hero (drift + source/prediction crossfade), editorial problem split, S1+DEM->U-Net->km2 pipeline diagram, lab workspace (scene rail + dominant viewport + conclusion panel), evidence section. Tokens: warm-black/ivory/teal #46bfa4/sand; Fraunces + Inter + Plex Mono.
- ui-ux-pro-max applied (swiss-minimal style, editorial type, subtle reveals, staged loading UX); neumorphism/neon rejected. All API/SSE contracts preserved; backend untouched.
- Validated live: Mekong 18.2236 / Ghana 0.0097 / Spain 13.005 / India 0.0 km2; 6 layers + export OK; JS syntax PASS; locked files untouched.

## T-SAO WEBSITE result (2026-10-08 10:35 UTC)
- Live demo at http://127.0.0.1:8000/ — FastAPI backend (website/backend/app.py) + dependency-free static frontend.
- All imagery/metrics from live M1 inference (SSE stages); verified Mekong 18.2236 / Ghana 0.01 / Spain 13.01 / India 0.00 km²; mask PNGs pixel-match predictions; export GeoTIFF georeferenced.
- 4 real scenes (event/date/coords from chips only). Locked files untouched. README documents architecture + startup.
