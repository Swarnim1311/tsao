# TSAO — Changelog (append-only)

## 2026-10-07 13:35 UTC — Project context system created
- Created `project_context/` with PROJECT_STATE, DECISIONS, CHANGELOG, DATASET_SPEC, NEXT_STEPS.
- Populated from verified NB01 results (25/25 gates PASS, executed 2026-10-07).
- Verified: 446 STAC catalog items → one event datetime per prefix; all 33 antecedent IMERG dates present among 36 files. DEM `nodata=None` (non-finite = invalid). IMERG lat ascending in these files (−89.95→89.95).

## 2026-10-07 — NB01_Dataset_Audit COMPLETE + LOCKED
- What: raw audit/EDA/QC over S1, labels, DEM, IMERG; official split audit; sample visualizations.
- Results: 446 chips (252/89/90/15); S1 512×512 VV/VH float32 dB EPSG:4326; labels {−1,0,1} aligned; flood 10.60% of valid px (9.16% of all; NoData 13.63%); per-chip flood% mean 10.87 / median 2.57; DEM 65 tiles readable; IMERG 36 files, `precipitation`, 1×1800×3600.
- Validation: 25/25 quality gates PASS (after fixing an S1-dtype string-vs-tuple comparison bug in the check itself — data was correct).
- Issues fixed: per-chip flood-% dict keyed by label name instead of S1 name (KeyError); dtype-check comparison bug. No raw data problems found.

## 2026-10-07 08:23 UTC - NB02_Multimodal_Preprocessing COMPLETE
- Processed 446 chips {'train': 252, 'valid': 89, 'test': 90, 'bolivia': 15} -> processed/{train,valid,test,bolivia}/*.npz
- Format: X(6,512,512) f32 [VV,VH,DEM,rain t-3/t-2/t-1], y(512,512) u8, valid_mask bool + metadata.csv/dataset_summary.json
- DEM: cached index, windowed mosaic, bilinear; multi-tile chips: 150
- IMERG: precipitation t-3/t-2/t-1 from STAC dates, explicit coords, bilinear; mean_valid_frac=0.862342
- Fill: S1 NaN->chip mean, DEM void->chip median, rain NaN->0.0; y zeroed where ~valid_mask (ignore-index)
- No normalization (train-only stats -> NB03). Validation: ALL PASS. NB02 safe to lock.

## 2026-10-07 09:15 UTC - NB03_Dataset_QC_and_Dataloaders COMPLETE
- Train-only stats over 252 train chips (valid_mask px, float64 accum): VV_dB mu=-10.449 sd=4.062, VH_dB mu=-17.316 sd=4.785, DEM_m mu=151.505 sd=137.359, rain_t-3 mu=7.070 sd=9.768, rain_t-2 mu=7.013 sd=8.435, rain_t-1 mu=14.304 sd=23.184
- Saved processed/normalization_stats.json (split=train, verified) + dataset_qc_summary.json
- FloodDataset (dict x/y/mask/chip_id; y float32; sanitize non-finite->0 post-norm) + loaders (batch=4, workers=0, pin=True, train shuffle seeded)
- QC: integrity 446/446, batch gates all splits, norm mean~0/std~1 on train, visual raw-vs-norm OK, leakage gates ALL PASS
- No training/augmentation/cropping/balancing; .npz untouched. NB03 safe to lock; next NB04 modeling.

## 2026-10-07 09:52 UTC - NB04_S1_UNet_Baseline COMPLETE
- UNet-32 (7,762,753 params) VV/VH->logits; masked 0.5BCE+0.5Dice; AdamW lr=0.001 wd=0.0001; batch=4; AMP=True on NVIDIA GeForce RTX 4050 Laptop GPU
- 32 epochs, 26.98 min; best ep 24 val IoU=0.6325 F1=0.7749 P=0.8149 R=0.7386
- TEST (once): IoU=0.6684 F1=0.8012 P=0.8144 R=0.7885
- ckpt models/nb04_s1_unet_best.pt + config; history results/nb04_training_history.csv; results/nb04_results.json
- Masked-loss invariance + overfit gates PASS. Bolivia untouched (loader smoke only). NB04 safe to lock; next NB05.

## 2026-10-07 10:05 UTC - NB04 post-run fixes + QC notes (append-only correction)
- Fixed 2 notebook code bugs found in executed output (no retraining needed; checkpoint/results unaffected):
  (a) §15 trivial assert touched wrapped dataset attr (S1Subset has no .paths) -> assert len(test)==90;
  (b) §16 normalized raw X with list.reshape -> np.array; selection now skips zero-valid chips for display.
- Verified fixed snippets by execution: test-eval assert passes; all 5 qualitative panels regenerated from saved best checkpoint.
- Data finding: 6 chips have valid_frac==0 (5 Ghana chips with all--1 labels = no ground truth; Paraguay_34417 test chip with 100%-NaN S1 but 100%-flood label). They contribute 0 pixels to loss/metrics by construction. Test metrics effectively cover 89 chips worth of pixels.
- Qualitative review (Ghana/Nigeria/Spain valid, Ghana/Paraguay test): strong TP on large flood bodies; under-detection of small scattered water (Spain); one FP blob (Paraguay_40936); invalid always grey. No background collapse. Credible baseline.

## 2026-10-07 10:32 UTC - NB05_Multimodal_UNet COMPLETE
- UNet-32 early-fusion 6ch (7,763,905 params, +1152 vs NB04); same seed/loss/opt/sched/patience/thr as NB04
- 28 epochs, 13.9 min; best ep 20 val IoU=0.6288 F1=0.7721
- TEST (once): IoU=0.6639 F1=0.7980 P=0.8899 R=0.7233
- vs NB04 locked (0.6684/0.8012): dIoU=-0.0045 dF1=-0.0032 (see results/nb04_vs_nb05_comparison.json)
- Gates (sanity/overfit/mask-invariance) PASS. Bolivia untouched. NB05 safe to lock; next NB06 ablations M0/M1/M2/M3.

## 2026-10-08 05:22 UTC - NB06_Controlled_Ablations COMPLETE
- Shared protocol, only first-conv differs (params M0 7,762,753 / M1 7,763,041 / M2 7,763,617 / M3 7,763,905)
- M1 S1+DEM: 14.83 min best ep 26 val IoU=0.6387; TEST IoU=0.6699 F1=0.8023 (d=+0.0015 vs M0)
- M2 S1+Rain: 8.24 min best ep 11 val IoU=0.6123; TEST IoU=0.6390 F1=0.7798 (d=-0.0294 vs M0)
- M0 locked 0.6684 / M3 locked 0.6639 cited, protocol-compatibility verified, not retrained
- Gates PASS; Bolivia untouched; 6 zero-valid chips retained. NB06 safe to lock; next NB07 Bolivia.

## 2026-10-08 05:23 UTC - NB06 pre-execution gate-calibration fix + execution notes (append-only correction)
- First nbconvert run STOPPED per failure rule at §10: M1 overfit gate failed narrowly (0.8711 -> 0.6069, ratio 0.697 vs required <0.60). No checkpoints/results were written by the failed run; locked M0/M3 untouched; notebook file unchanged by the failed run.
- Read-only diagnostic (seed-42 replica, script outside project tree): M1 crosses 0.60 at step 132/150, M2 at step 91/150; both loss curves smooth and monotonic. Verdict: gate miscalibrated (60 steps too few for full-res 512px UNet overfit), not a model defect. User-approved minimal fix: §10 overfit loop 60 -> 150 steps, threshold <0.60 unchanged; only 2 lines changed in NB06_Controlled_Ablations.ipynb (verified by diff vs backup).
- Second nbconvert run hit a 600 s per-cell nbconvert timeout mid-M1-training (partial M1 checkpoint overwritten deterministically on re-run); re-ran with per-cell timeout disabled -> full execution SUCCESS (22/22 cells, zero errors).
- Post-execution validation (independent script): M1/M2 checkpoints load with exact param counts; histories match epochs/best-epoch/best-val; M0/M3 values unchanged; master ablation file present.

## 2026-10-08 05:38 UTC - NB07_Bolivia_Generalization COMPLETE (inference-only)
- 15 Bolivia chips, valid_px=2,824,434, zero-valid=0
- Bolivia IoU: M0=0.6167, M1=0.6311, M2=0.5479, M3=0.3636
- Gaps (main-bolivia IoU): M0=+0.0517, M1=+0.0388, M2=+0.0911, M3=+0.3003
- Locked ckpts verified by params+hash; no training/tuning; thr 0.5; train-only norm. NB07 safe to lock.

## 2026-10-08 05:47 UTC - NB08_Operating_Point_and_Error_Analysis COMPLETE (analysis-only)
- Valid-selected IoU thresholds: M0=0.4, M1=0.5, M2=0.5, M3=0.45
- M3 Bolivia @valid-thr vs @0.5 vs M0@0.5: see nb08_error_analysis.csv; threshold verdict in notebook §16
- M3 failure mode: recall collapse (see error table); distribution comparison in nb08_distribution_shift.json
- M1: moderate geographic drop (0.0388) but strongest/most robust on Bolivia; ±0.005 band is single-run variation, NOT geographic shift (binding clarification)
- No training/tuning; locked results preserved. NB08 safe to lock.

## 2026-10-08 07:48 UTC - NB09_Calibration_SmallWater_Multiseed COMPLETE
- Calibration (valid+Bolivia, locked ckpts): see nb09_calibration_summary.csv (Brier/ECE descriptive)
- Small-water morphology: GT-component recall by bin valid/test/Bolivia + FP morphology CSVs
- Multiseed (42 locked + 7/123 trained, identical protocol): see nb09_multiseed_summary.csv
- Preferred model + architecture verdict: see nb09_results.json (conservative; no significance claims)
- Locked files unchanged; seed42 never retrained; Bolivia/test never tuned. NB09 safe to lock.

## 2026-10-08 08:24 UTC - NB10_Attention_UNet_Final_Model COMPLETE
- Attention U-Net (3ch VV/VH/DEM, 7,893,845 params) vs locked M1: 13.5 min, best ep 19, val IoU=0.6182
- TEST IoU=0.6606 F1=0.7956 (d=-0.0093 vs M1)
- BOLIVIA IoU=0.5004 F1=0.6670 (d=-0.1308 vs M1)
- FINAL MODEL = M1 (rule: test>=M1+0.005 AND Bolivia>=M1-0.01). NB10 safe to lock.

## 2026-10-08 09:43 UTC - FINAL_INFERENCE_PIPELINE COMPLETE
- Reusable module src/inference/ (unet.py locked arch; flood_predictor.py FloodPredictor/CLI/GeoTIFF/viz/area; __main__.py).
- Preprocessing exactly NB03/NB06: train-stats (x-mean)/std on VV/VH/DEM idx [0,1,2], non-finite->0, sigmoid, thr 0.5, valid_mask respected. No rainfall, no training, no tuning.
- Test reproduction (90 chips): IoU=0.669888 F1=0.802315 P=0.818482 R=0.786774 vs locked 0.6699/0.8023/0.8185/0.7868 (diff ~1e-6, AMP float; PASS).
- Area: per-pixel WGS84 latitude-corrected (~97-98 m2/px, EPSG:4326, never a constant). Mekong_1443339: GT 18.7210 vs pred 18.2236 km2, abs 0.4974, rel 2.66% (descriptive).
- GeoTIFFs predictions/probabilities|masks/Mekong_1443339_m1_{prob,mask}.tif (EPSG:4326, transform preserved); panels results/final_inference_examples/{Mekong_1443339,Ghana_1078550}[_NB11]_panel.png.
- CLI: python -m src.inference --npz <chip> --output <dir> [--save-viz] (+ trio --vv/--vh/--dem mode). NB11 demo notebook executed 0 errors.
- models/final_model_config.json created (M1, test 0.6699/0.8023/0.8185/0.7868, Bolivia 0.6311/0.7739/0.8899/0.6846, seed 42).
- 22/22 quality gates PASS. Locked files untouched (446 npz, 13 ckpts, NB01-NB10). D19 recorded. NEXT STEP = T-SAO WEBSITE.

## 2026-10-08 10:35 UTC - T-SAO_WEBSITE COMPLETE
- Stack: FastAPI backend (website/backend/app.py + layers.py PIL renderer) serving API + static frontend (index.html/styles.css/app.js, no framework/build) on :8000.
- Endpoints: /api/health, /api/model-info, /api/scenes (real chip metadata), POST /api/predict/{scene} (SSE real stages, live FloodPredictor), /api/scenes/{scene}/layer/{vv,vh,dem,prob,mask,gt} PNGs, /api/scenes/{scene}/export mask GeoTIFF.
- Design: near-black charcoal + off-white + single teal flood accent; Inter/Plex Mono; hero source/prediction crossfade; capability strip; 5-mode canvas (source/prob/mask/GT/compare slider) with zoom-pan; count-up km²; staged SSE loading; friendly errors; keyboard tabs/slider/scenes; reduced-motion; responsive incl. tablet.
- Validated live: Mekong 18.2236 / Ghana 0.01 / Spain 13.01 / India 0.00 km²; mask PNG teal-px == flooded_pixels; export 187661 px + EPSG:4326 transform; 404 friendly; zero console errors; screenshots reviewed.
- Fixes: double-inference generator bug (rewrote SSE stream); Ghana small-mask assertion (size→content check); compare slider thumb styling.
- Locked files untouched (446 npz, 13 ckpts, 12 notebooks, norm stats). D20 recorded. README added. Server left running for review. NEXT STEP = FINAL PROJECT POLISH / REVIEW PREPARATION.

## 2026-10-08 11:30 UTC - T-SAO_VISUAL_REDESIGN COMPLETE
- Full frontend recomposition (website/frontend only): cinematic full-viewport SAR hero with slow drift + source/prediction crossfade (replaces headline-left/image-right layout); editorial problem split; visual S1+DEM->U-Net->km2 pipeline diagram; immersive lab workspace (scene rail + dominant viewport + conclusion panel); evidence section (count-up locked metrics + ablation table).
- Design tokens: warm-black #0d0f0e, ivory #ede9df, single aquatic accent #46bfa4, mineral sand secondary; Fraunces display serif + Inter UI + IBM Plex Mono technical labels.
- ui-ux-pro-max skill applied: minimalism-swiss style (grid/whitespace/single accent), News-Editorial serif+sans+mono pairing, subtle-tier reveals (300-700ms, y~14px), skeleton+staged progress UX, sliding tab indicator; auto-suggested neumorphism + OLED-neon directions explicitly rejected per brief (accessibility risk, wrong register).
- Interactions added: fixed nav with scrolled blur + active-section indicator, per-scene metadata panel, 4-stage inference stepper, mode sliding ink, view captions, compare centre-line + handle, evidence count-up; all timings 150-700ms; prefers-reduced-motion fully respected.
- Contracts preserved: all backend endpoints + SSE stages unchanged; backend app.py/layers.py untouched; all legacy frontend element IDs retained and extended only.
- Validated live: Mekong 18.2236 / Ghana 0.0097 / Spain 13.005 / India 0.0 km2; all 6 layers + GeoTIFF export OK; node --check app.js PASS; zero missing element IDs; locked model/npz/norm-stats/inference module untouched.
- D21 recorded. Server left running for review. NEXT STEP = FINAL PROJECT POLISH / REVIEW PREPARATION.

## 2026-10-08 12:10 UTC - T-SAO_COMMAND_REDESIGN COMPLETE (major visual redesign)
- Replaced editorial/Fraunces direction with futuristic flood-intelligence command: Space Grotesk display + Inter body + JetBrains Mono technical (ui-ux-pro-max Web3-DeFi tri-stack, verified in skill data).
- Palette rebuilt: graphite/smoky-navy base, electric aqua #3ee6c4 primary/flood, lime #b6f34a verified, amber warning, red-orange critical — used semantically only. HUD thin-lines + liquid-glass-selective per skill.
- images/img1.jpg (ocean wave) integrated as fixed full-page environment: 6.6MB to 1920px/335KB website/frontend/bg.jpg, deep-navy scrims + data-grid + scan sweep + slow drift + pointer parallax.
- Layout: floating pill nav, hero (identity left / live-status glass panel right / HUD strip bottom), glass SCENE/MAP/DATA workstation deck, hover-interactive 5-stage pipeline, animated ablation bars + count-up evidence.
- Interactions: parallax, magnetic buttons, CSS tooltips, sliding mode ink, compare handle with SOURCE/FLOOD tags, mask reveal on inference, deck IDLE/PROCESSING/COMPLETE, 18-point checklist all covered, reduced-motion + reduced-transparency fallbacks.
- Validated live with zero console errors: Mekong 18.2236 / Ghana 0.0097; all IDs present; node --check PASS; 5 Playwright screenshots reviewed (website/tsao_{hero,analyze,compare}.png); backend + inference + locked files untouched.
- D22 recorded. Server left running for review. NEXT STEP = FINAL PROJECT POLISH / REVIEW PREPARATION.

## 2026-10-08 13:00 UTC - T-SAO_PREMIUM_REFINEMENT COMPLETE
- Background swapped to images/img2.jpg (satellite estuary): optimized to website/frontend/bg2.jpg (8375px/11.7MB to 1920px/736KB), retuned scrims so terrain/water texture breathes while text stays readable; preload updated.
- Full name wired in: hero hairline (Terrain-Sensitive Assessment of Overflow), method line, rebuilt footer (brand + full name + model/data/mode meta + locked base strip), title tag, wordmark tooltip.
- Navbar premium: entrance slide-fade, sliding gradient underlines (hover .7 / active full + glow), wordmark icon glow + tracking shift, pressed states, stronger scrolled blur/glow, links kept visible on mobile (dot-only status under 560px).
- UX: plain-language flood verdict from live fraction (No/Minimal/Localized/Significant/Major + % of valid scene), "Open compare" next-action shortcut, standby empty state with reticle (skill: helpful empty states), staggered scene-card entrances, deck top sheen, gradient primary buttons with inset highlight.
- Skill queries run (navbar active-state, empty-state guidance) and applied; no backend/model/inference changes; no fake data (verdict derives from live flood_fraction).
- Validated: Mekong 18.2236 / Ghana 0.0097, zero console errors, bg2 + all IDs + full-name probes PASS, evidence settles to 0.6699/0.8023/0.6311, reduced-motion reveals all visible, desktop/tablet/mobile screenshots reviewed (website/tsao2_*.png). D23 recorded.

## 2026-10-08 13:40 UTC - FINAL_REPOSITORY_AUDIT COMPLETE (submission-readiness pass)
- Verified production artifacts: final_model_config (test 0.6699/0.8023/0.8185/0.7868, Bolivia 0.6311/0.7739/0.8899/0.6846, thr 0.5, seed 42, 3ch) consistent with nb06 ablation + nb07 + nb09 + nb10 files; checkpoint param assert (7763041) passes at load; no contradictions.
- Inference smoke (direct FloodPredictor): Mekong 18.2236 (IoU 0.9102), Ghana 0.0097 — exact locked reproduction. Pipeline asserts (threshold/stats/channels/finite/mask) intact.
- Website E2E (Playwright, zero console errors): Mekong analysis settles 18.22 + Major verdict, Ghana 0.01 + Minimal verdict, compare line works, flood mask renders real trace-flood darkness for Ghana, nav works, brand line renders.
- Secrets scan (src/website/notebooks): clean — only benign prose matches.
- Git: .gitignore extended (website/.cache, .kilo, images/img2.jpg source, website/*.png shots); experimental ~560MB checkpoints correctly excluded; tracked tree ~41MB (31MB M1 + research).
- Removed dead tracked assets: website/frontend/bg.jpg, website/tsao_*.png (superseded), images/img1.jpg (unreferenced) — all unreferenced in code.
- Rewrote README.md into 19-section research README with locked values only, honest limitations, no invented citations/claims.
- NB01–NB11 present, no backups/checkpoints; results figures present; absolute local paths exist only inside historical results JSON metadata (not code).
- D24 recorded. NOT committed — awaiting manual git commit/push by user.
