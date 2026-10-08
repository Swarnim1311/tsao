"""T-SAO demo backend: FastAPI + real src.inference FloodPredictor.

No model logic is duplicated here: normalization, thresholding, masking and
area estimation all run inside src.inference.flood_predictor.FloodPredictor.
"""
import asyncio
import io
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from layers import (
    gray_layer, dem_layer, prob_layer, mask_layer, gt_layer, to_png_bytes,
)

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from src.inference import FloodPredictor, compute_metrics  # noqa: E402
from src.inference.flood_predictor import transform_to_affine  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / ".cache"
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

SCENES = ["Mekong_1443339", "Ghana_1078550", "Spain_7387658", "India_44475"]
SPLIT = "test"

app = FastAPI(title="T-SAO Flood Mapping Demo")
predictor = FloodPredictor()
model_cfg = json.loads((ROOT / "models" / "final_model_config.json").read_text())


def _chip_meta(chip_id):
    p = ROOT / "processed" / SPLIT / f"{chip_id}.npz"
    if not p.exists():
        raise HTTPException(status_code=404, detail="This scene isn't compatible with the current model.")
    d = np.load(p, allow_pickle=True)
    b = [float(v) for v in np.asarray(d["bounds"])]
    vm = d["valid_mask"].astype(bool)
    y = d["y"].astype(np.uint8)
    nv = int(vm.sum())
    return {
        "scene": chip_id,
        "event": str(d["event"]),
        "event_date": str(d["event_date"]),
        "split": str(d["split"]),
        "centre_lat": round((b[1] + b[3]) / 2, 4),
        "centre_lon": round((b[0] + b[2]) / 2, 4),
        "valid_pixels": nv,
        "gt_flood_pixels": int(((y == 1) & vm).sum()) if nv else 0,
        "has_ground_truth": bool(nv > 0),
    }


def _run_inference(chip_id):
    """Real inference. Returns (result_dict, arrays_dict). Raises RuntimeError with friendly msg."""
    p = ROOT / "processed" / SPLIT / f"{chip_id}.npz"
    if not p.exists():
        raise RuntimeError("Terrain data is unavailable for this scene.")
    try:
        out = predictor.predict_npz(str(p))
    except Exception:
        raise RuntimeError("Flood analysis couldn't be completed.")
    d = np.load(p, allow_pickle=True)
    X = d["X"].astype(np.float32)
    bench = compute_metrics(out["mask"], out["gt"], out["valid_mask"]) if int(out["valid_mask"].sum()) else None
    result = {
        "scene": chip_id,
        "flood_area_km2": round(float(out["flooded_area_km2"]), 4),
        "flood_fraction": round(float(out["flood_fraction"]), 6),
        "flooded_pixels": int(out["flooded_pixel_count"]),
        "valid_pixels": int(out["valid_pixel_count"]),
        "threshold": 0.5,
        "model": "T-SAO M1",
        "architecture": "U-Net-32 + Sentinel-1 VV/VH + DEM",
        "benchmark": ({k: round(float(v), 4) for k, v in bench.items()
                       if k in ("iou", "f1", "precision", "recall")} if bench else None),
        "layers": {k: f"/api/scenes/{chip_id}/layer/{k}"
                   for k in ("vv", "vh", "dem", "prob", "mask", "gt")},
        "export_geotiff": f"/api/scenes/{chip_id}/export",
    }
    arrays = {"vv": X[0], "vh": X[1], "dem": X[2], "prob": out["probability"],
              "mask": out["mask"], "gt": out["gt"], "valid": out["valid_mask"],
              "transform": np.asarray(d["transform"]), "crs": str(d["crs"])}
    return result, arrays


def _cache_result(chip_id, result):
    cdir = CACHE / chip_id
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def _render_layer(chip_id, kind, arrays=None):
    cdir = CACHE / chip_id
    cdir.mkdir(parents=True, exist_ok=True)
    lp = cdir / f"{kind}.png"
    if lp.exists():
        return lp.read_bytes()
    if arrays is None:
        _, arrays = _run_inference(chip_id)
    v = arrays["valid"]
    if kind == "vv":
        img = gray_layer(arrays["vv"], v)
    elif kind == "vh":
        img = gray_layer(arrays["vh"], v)
    elif kind == "dem":
        img = dem_layer(arrays["dem"], v)
    elif kind == "prob":
        img = prob_layer(arrays["prob"], v)
    elif kind == "mask":
        img = mask_layer(arrays["mask"], v)
    elif kind == "gt":
        img = gt_layer(arrays["gt"], v)
    else:
        raise HTTPException(status_code=404, detail="Unknown visualization layer.")
    data = to_png_bytes(img)
    lp.write_bytes(data)
    return data


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "M1", "device": str(predictor.device)}


@app.get("/api/model-info")
def model_info():
    return model_cfg


@app.get("/api/scenes")
def scenes():
    return [_chip_meta(c) for c in SCENES]


@app.post("/api/predict/{chip_id}")
async def predict(chip_id: str):
    if chip_id not in SCENES:
        raise HTTPException(status_code=404, detail="This scene isn't compatible with the current model.")

    async def stream():
        def emit(obj):
            return ("data: " + json.dumps(obj) + "\n\n").encode()
        yield emit({"stage": "preparing", "label": "Preparing scene"})
        await asyncio.sleep(0)
        try:
            result, arrays = await asyncio.to_thread(_run_inference, chip_id)
        except RuntimeError as e:
            yield emit({"error": str(e)})
            return
        yield emit({"stage": "normalizing", "label": "Normalizing inputs"})
        yield emit({"stage": "segmenting", "label": "Running segmentation"})
        yield emit({"stage": "extent", "label": "Calculating flood extent"})
        _cache_result(chip_id, result)
        for kind in ("vv", "vh", "dem", "prob", "mask", "gt"):
            await asyncio.to_thread(_render_layer, chip_id, kind, arrays)
        yield emit({"stage": "done", "label": "Complete", "result": result})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/scenes/{chip_id}/layer/{kind}")
def layer(chip_id: str, kind: str):
    if chip_id not in SCENES:
        raise HTTPException(status_code=404, detail="This scene isn't compatible with the current model.")
    try:
        data = _render_layer(chip_id, kind)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StreamingResponse(io.BytesIO(data), media_type="image/png")


@app.get("/api/scenes/{chip_id}/export")
def export(chip_id: str):
    if chip_id not in SCENES:
        raise HTTPException(status_code=404, detail="This scene isn't compatible with the current model.")
    cdir = CACHE / chip_id
    cdir.mkdir(parents=True, exist_ok=True)
    out = cdir / f"{chip_id}_m1_mask.tif"
    if not out.exists():
        result, arrays = _run_inference(chip_id)
        _cache_result(chip_id, result)
        import rasterio
        from rasterio.crs import CRS
        from affine import Affine
        t = list(arrays["transform"])
        aff = Affine(t[0], t[1], t[2], t[3], t[4], t[5]) if len(t) == 9 else Affine(*t[:6])
        with rasterio.open(out, "w", driver="GTiff", height=512, width=512, count=1,
                           dtype="uint8", crs=CRS.from_string(arrays["crs"]),
                           transform=aff, compress="deflate") as dst:
            dst.write(arrays["mask"].astype(np.uint8), 1)
    return FileResponse(out, media_type="image/tiff",
                        filename=f"{chip_id}_m1_flood_mask.tif")


app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
