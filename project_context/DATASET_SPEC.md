# TSAO — Canonical Dataset Spec

**Last updated:** 2026-10-08 05:47 UTC (NB08 analyzed; spec unchanged)

## Splits (official, frozen)
| Split | CSV | Chips |
|---|---|---|
| train | `splits/flood_handlabeled/flood_train_data.csv` | 252 |
| valid | `.../flood_valid_data.csv` | 89 |
| test | `.../flood_test_data.csv` | 90 |
| bolivia (holdout) | `.../flood_bolivia_data.csv` | 15 |
| total | | 446 |
CSV: no header, `S1Hand_filename,LabelHand_filename` per line. Zero within-split duplicates, zero cross-split overlap, zero missing/unreferenced.

## Sentinel-1 (S1Hand)
- Path: `data/flood_events/HandLabeled/S1Hand/<Event>_<chipid>_S1Hand.tif`
- 512×512, 2 bands [VV, VH], float32, EPSG:4326, res ≈ 8.9831528e-05° (~10 m), dB backscatter.
- NaN = NoData (~2.75% in 30-chip sample). Sample stats: VV min −49.05 / max 25.60 / mean −10.44 / std 3.95; VH min −56.69 / max 10.48 / mean −17.59 / std 5.05.
- Canonical grid: exact shape/transform/CRS/bounds preserved; never reprojected.

## Flood labels (LabelHand)
- Path: `data/flood_events/HandLabeled/LabelHand/<Event>_<chipid>_LabelHand.tif`
- 512×512, 1 band (int16 on disk), EPSG:4326, same grid as S1 (verified aligned — never resample).
- Values: −1 NoData (13.63%), 0 non-flood (77.22%), 1 flood (9.16%); no other values. Flood = 10.6014% of valid pixels (100,983,314 valid / 15,932,910 NoData).
- Per-chip flood% (446 chips): mean 10.87, median 2.57, std 19.15, min 0.0, max 100.0. Split mean flood%: train 9.38 / valid 11.42 / test 13.51 / bolivia 16.99.

## Copernicus DEM (GLO-30)
- Root: `C:/FloodProject/DEM`; 65 main files `Copernicus_DSM_COG_10_*_DEM.tif` (recursive, one folder per tile).
- 3600×3600, float32, EPSG:4326, res 0.0002777778° (~30 m); metres; `nodata` attr None → non-finite = invalid. All readable. Sampled elev range ≈ −4…1515 m.

## NASA IMERG (GPM Final Run V07B, daily)
- Root: `C:/FloodProject/IMERG/data`; 36 × `3B-DAY.MS.MRG.3IMERG.<YYYYMMDD>-S000000-E235959.V07B.nc4`.
- Variable `precipitation` only (mm/day); dims time=1, lat=1800, lon=3600; 0.1° global grid (lon −179.95…179.95 asc; lat −89.95…89.95 **ascending** in these files — detect, don't assume).
- Antecedent mapping (event → required t−3/t−2/t−1, all present):
  - India 2016-08-12 → 08-09/10/11; Sri-Lanka 2017-05-30 → 05-27/28/29; Pakistan 2017-06-28 → 06-25/26/27;
  - Bolivia 2018-02-15 → 02-12/13/14; Somalia 2018-05-07 → 05-04/05/06; Mekong 2018-08-05 → 08-02/03/04;
  - Ghana 2018-09-18 → 09-15/16/17; Nigeria 2018-09-21 → 09-18/19/20; Paraguay 2018-10-31 → 10-28/29/30;
  - USA 2019-05-22 → 05-19/20/21; Spain 2019-09-17 → 09-14/15/16.
- Per-chip event date: `catalog/sen1floods11_hand_labeled_source/<Event>_<chipid>.json` → `properties.datetime` (446 items, one date per event prefix; `Mekong` = Cambodia event).

## Events (11)
Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA.

## NB02 processed spec (CONFIRMED on disk 2026-10-07, 13/13 validation PASS)
- `processed/{train,valid,test,bolivia}/<chip>.npz`: `X` (6,512,512) float32 [VV,VH,DEM,rain t−3,t−2,t−1], `y` (512,512) uint8 {0,1}, `valid_mask` (512,512) bool, metadata scalars. + `metadata.csv`, `dataset_summary.json`.

## Caveats
- Chip footprint (~0.046°) < one IMERG cell (0.1°): rainfall channels are smooth antecedent context, not fine-scale data.
- Imbalanced: median chip only ~2.6% flood → handle in loss/metrics later.

Confirmed: 446 .npz (train 252 / valid 89 / test 90 / bolivia 15), X(6,512,512) f32 finite, y u8 (0/1 on valid, 0 where masked), valid_mask bool, mean valid_frac 0.862342, 150 multi-tile chips.

Zero-valid chips (6, contribute 0 px to loss/metrics): 5 Ghana chips with all-NoData labels (Ghana_234935/D277/D5079 valid; Ghana_26376 train; Ghana_83483 test) + Paraguay_34417 (test, S1 100% NaN, label 100% flood valid-px). Test metrics effectively aggregate over 89 chips.

## NB06 re-verification (2026-10-08 05:23 UTC, no spec change)
Split counts re-asserted at NB06 runtime (train 252 / valid 89 / test 90 / bolivia 15); train-only normalization stats reused unchanged; 6 zero-valid chips retained in loaders; Bolivia never trained on.

## NB07 Bolivia evaluation (2026-10-08 05:38 UTC, no spec change)
15 Bolivia chips evaluated inference-only (valid_px=2,824,434; flood_px=451,901; zero-valid Bolivia chips: 0 — all 15 contribute). Split integrity re-asserted (zero overlap); train-only normalization reused; per-chip flood fractions range 0.000007–0.658 (median ≈ 0.061).

## NB08 distribution analysis (2026-10-08 05:47 UTC, no spec change)
Raw-physical valid-px stats, test vs Bolivia: rain_t-3 median 2.24 vs 18.20 mm/d; rain_t-1 median 0.80 vs 12.70 mm/d; rain zero-fraction ~0.21–0.23 (test) vs 0.0 (Bolivia, no dry pixels); DEM median 109 vs 147 m (modest shift). Bolivia antecedent rainfall is substantially wetter — relevant context for rain-fed models (M2/M3); association only, not causality.
