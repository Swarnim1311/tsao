# T-SAO — Pixel-level flood mapping from Sentinel-1 SAR

Deep-learning flood mapping from Sentinel-1 SAR (VV/VH) + Copernicus DEM elevation.
Final locked model **M1**: U-Net-32, inputs VV/VH/DEM, threshold 0.50.

- Test IoU **0.6699**, F1 **0.8023** · Bolivia holdout IoU **0.6311**
- Canonical grid: 512×512, EPSG:4326 · Train-only normalization · No rainfall

## Repository layout

| Path | Contents |
|---|---|
| `website/frontend/` | Static demo site (`index.html`, `styles.css`, `app.js`) — no build step |
| `website/backend/` | FastAPI demo server (`app.py`, `layers.py`) — calls real `FloodPredictor` |
| `src/inference/` | Locked production inference (`FloodPredictor`, UNet-32, CLI, area, GeoTIFF) |
| `models/nb06_m1_s1_dem_best.pt` | Locked M1 checkpoint (do not modify) |
| `models/final_model_config.json` | Locked model configuration |
| `processed/` | 446 per-chip `.npz` + train-derived `normalization_stats.json` (do not modify) |
| `notebooks/NB01–NB11` | Locked research notebooks (do not modify) |

## Website architecture

```
Browser (static HTML/CSS/JS, no framework)
  │  GET /api/scenes · POST /api/predict/{scene} (SSE stages) · GET /api/scenes/{scene}/layer/{kind} · GET /api/scenes/{scene}/export
  ▼
FastAPI (website/backend/app.py, port 8000)
  │  FloodPredictor (src/inference/flood_predictor.py) — normalization, U-Net,
  │  threshold 0.5, valid_mask, latitude-corrected WGS84 area. No logic duplicated.
  ▼
PNG layers (website/.cache) + result JSON + mask GeoTIFF export
```

No model logic lives in the frontend. Every displayed prediction comes from live
M1 inference. Displayed test/Bolivia metrics are the locked NB06/NB07 values.

## Start the demo

Backend (serves API + frontend on one port):

```powershell
cd website/backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/>.

Standalone inference CLI (no website needed):

```powershell
python -m src.inference --npz processed/test/Mekong_1443339.npz --output predictions/demo --prefix demo --save-viz
```

## Demo scenes (all real `processed/test` chips with ground truth)

| Scene | Event date | Character |
|---|---|---|
| Mekong_1443339 | 2018-08-05 | Major flood, 18.22 km² predicted |
| Ghana_1078550 | 2018-09-18 | Minimal water, 0.01 km² predicted |
| Spain_7387658 | 2019-09-17 | Large flood, 13.01 km² predicted |
| India_44475 | 2016-08-12 | No flood, 0.00 km² predicted |

Scene metadata (event, date, centre coordinates) is read live from the processed
chips — nothing is invented. Benchmark IoU/F1 shown beside live predictions are
computed against the real labels at inference time.

## Inference workflow (website)

1. Select a scene → source imagery loads (real VV/VH/DEM renders).
2. **Analyze scene** → server streams real stages (prepare → normalize →
   segment → extent) over SSE while `FloodPredictor` runs on GPU.
3. Visualization reveals: source / probability / flood mask / ground truth /
   draggable source-vs-prediction comparison, with zoom/pan.
4. Results count up: flood extent km², coverage, pixel counts, benchmark
   metrics (clearly labelled ground-truth evaluation), GeoTIFF + PNG export.
"# tsao" 
