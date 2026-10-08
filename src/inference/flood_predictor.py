"""Final M1 production inference pipeline (LOCKED model, no training).

Model: U-Net-32, inputs VV/VH/DEM (3ch), train-derived standardization,
sigmoid + threshold 0.5, valid_mask respected.

Preprocessing is EXACTLY NB03/NB06:
    x_norm = (x_physical - mean_train) / std_train   (per channel, float32)
    x_norm[~isfinite] -> 0.0                         (== channel mean)
Channel order preserved: VV=X[0], VH=X[1], DEM=X[2] (indices [0,1,2] of NB02 6ch).
No rainfall is ever used.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from .unet import UNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "nb06_m1_s1_dem_best.pt"
DEFAULT_STATS = PROJECT_ROOT / "processed" / "normalization_stats.json"

MODEL_NAME = "M1"
INPUT_CHANNEL_NAMES = ["VV_dB", "VH_dB", "DEM_m"]
INPUT_CHANNEL_IDX = [0, 1, 2]  # positions inside NB02 6ch X
FULL6 = ["VV_dB", "VH_dB", "DEM_m", "rain_t-3", "rain_t-2", "rain_t-1"]
THRESHOLD = 0.5
EXPECTED_PARAMS = 7763041
EPS = 1e-9

# WGS84 approx (meters per degree as a function of latitude, degrees).
def _m_per_deg_lat(lat_deg):
    p = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    return 111132.92 - 559.82 * np.cos(2 * p) + 1.175 * np.cos(4 * p) - 0.0023 * np.cos(6 * p)

def _m_per_deg_lon(lat_deg):
    p = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    return 111412.84 * np.cos(p) - 93.5 * np.cos(3 * p) + 0.118 * np.cos(5 * p)


def pixel_area_map_m2(shape, transform):
    """Per-pixel area (m^2) for an EPSG:4326 grid.

    transform: rasterio Affine (a,b,c,d,e,f) or 6/9-element sequence.
    Pixel height/width in degrees come from the transform; latitude of each
    row centre sets the local meters-per-degree (WGS84). Returns (H,W) float64.
    """
    from affine import Affine
    if isinstance(transform, Affine):
        aff = transform
    else:
        t = list(transform)
        if len(t) == 9:  # 3x3 flattened as stored in processed/*.npz
            t = [t[0], t[1], t[2], t[3], t[4], t[5]]
        assert len(t) == 6, f"transform must be Affine or 6/9 numbers, got {len(t)}"
        aff = Affine(*t)
    H, W = shape
    pix_w_deg = abs(aff.a)
    pix_h_deg = abs(aff.e)
    assert aff.b == 0 and aff.d == 0, "rotated transforms not supported"
    rows = np.arange(H, dtype=np.float64)
    lat_centre = aff.f + (rows + 0.5) * aff.e  # aff.e < 0 for north-up
    row_area = _m_per_deg_lat(lat_centre) * pix_h_deg * _m_per_deg_lon(lat_centre) * pix_w_deg
    return np.repeat(row_area[:, None], W, axis=1)


def transform_to_affine(transform):
    from affine import Affine
    if isinstance(transform, Affine):
        return transform
    t = list(transform)
    if len(t) == 9:
        t = [t[0], t[1], t[2], t[3], t[4], t[5]]
    return Affine(*t)


def compute_metrics(pred_mask, gt, valid_mask):
    """Mask-respecting binary metrics (identical to NB04/NB06 summarize)."""
    pred = np.asarray(pred_mask).astype(bool)
    t = (np.asarray(gt).astype(np.float32) > 0.5)
    m = np.asarray(valid_mask).astype(bool)
    tp = float(np.sum(pred & (t == 1) & m))
    fp = float(np.sum(pred & (t == 0) & m))
    fn = float(np.sum((~pred) & (t == 1) & m))
    tn = float(np.sum((~pred) & (t == 0) & m))
    iou = tp / (tp + fp + fn + EPS)
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    f1 = 2 * prec * rec / (prec + rec + EPS)
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


class FloodPredictor:
    """Reusable M1 inference. No training occurs here."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, stats_path=DEFAULT_STATS,
                 threshold=THRESHOLD, device=None):
        assert float(threshold) == 0.5, f"threshold locked at 0.5, got {threshold}"
        self.threshold = 0.5
        self.checkpoint_path = Path(checkpoint_path)
        self.stats_path = Path(stats_path)
        assert self.checkpoint_path.exists(), f"missing checkpoint {self.checkpoint_path}"
        assert self.stats_path.exists(), f"missing stats {self.stats_path}"
        st = json.loads(self.stats_path.read_text())
        assert st.get("split") == "train", "normalization stats must be train-derived"
        assert st.get("channels") == FULL6, f"unexpected channels {st.get('channels')}"
        full_mean = np.array(st["mean"], dtype=np.float32)
        full_std = np.array(st["std"], dtype=np.float32)
        assert (full_std > 0).all()
        self.mean = full_mean[INPUT_CHANNEL_IDX].reshape(-1, 1, 1)
        self.std = full_std[INPUT_CHANNEL_IDX].reshape(-1, 1, 1)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = UNet(in_ch=3, base=32).to(self.device)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state)
        n_params = sum(p.numel() for p in self.model.parameters())
        assert n_params == EXPECTED_PARAMS, f"arch/checkpoint mismatch: {n_params}"
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def preprocess(self, vv, vh, dem):
        """Validate + standardize with train stats. Returns (3,H,W) float32 normalized."""
        vv = np.asarray(vv, dtype=np.float32)
        vh = np.asarray(vh, dtype=np.float32)
        dem = np.asarray(dem, dtype=np.float32)
        assert vv.shape == vh.shape == dem.shape and vv.ndim == 2, \
            f"shapes must match (H,W), got {vv.shape} {vh.shape} {dem.shape}"
        assert min(vv.shape) > 0
        x = np.stack([vv, vh, dem], axis=0).astype(np.float32)  # (3,H,W) physical
        xn = (x - self.mean) / self.std
        xn = np.where(np.isfinite(xn), xn, 0.0).astype(np.float32)  # NB03 sanitize
        assert np.isfinite(xn).all()
        return xn

    @torch.no_grad()
    def predict(self, vv, vh, dem, valid_mask=None, transform=None, crs="EPSG:4326"):
        """Run full inference. Returns dict(prob, mask, valid_mask, counts, fraction, areas)."""
        vv_a = np.asarray(vv, dtype=np.float32)
        shape = vv_a.shape
        if valid_mask is None:
            finite = np.isfinite(np.asarray(vv, dtype=np.float32)) & \
                np.isfinite(np.asarray(vh, dtype=np.float32)) & \
                np.isfinite(np.asarray(dem, dtype=np.float32))
            vm = finite
        else:
            vm = np.asarray(valid_mask).astype(bool)
            assert vm.shape == shape, f"valid_mask shape {vm.shape} != input {shape}"
            finite = np.isfinite(np.asarray(vv, dtype=np.float32)) & \
                np.isfinite(np.asarray(vh, dtype=np.float32)) & \
                np.isfinite(np.asarray(dem, dtype=np.float32))
            vm = vm & finite  # never score non-finite pixels
        xn = self.preprocess(vv, vh, dem)
        xt = torch.from_numpy(xn).unsqueeze(0).to(self.device)
        assert tuple(xt.shape[1:]) == (3, shape[0], shape[1])
        logits = self.model(xt).float()
        assert torch.isfinite(logits).all(), "non-finite logits"
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
        assert np.isfinite(prob).all() and prob.min() >= 0.0 and prob.max() <= 1.0
        mask = (prob >= self.threshold).astype(np.uint8)
        mask[~vm] = 0  # valid_mask respected
        valid_count = int(vm.sum())
        flooded_count = int((mask == 1).sum())
        flood_fraction = (flooded_count / valid_count) if valid_count else 0.0
        if transform is not None:
            area_map = pixel_area_map_m2(shape, transform)
            flooded_area_m2 = float(area_map[(mask == 1)].sum())
        else:
            area_map = None
            flooded_area_m2 = float("nan")
        return {
            "probability": prob,
            "mask": mask,
            "valid_mask": vm,
            "valid_pixel_count": valid_count,
            "flooded_pixel_count": flooded_count,
            "flood_fraction": float(flood_fraction),
            "flooded_area_m2": float(flooded_area_m2),
            "flooded_area_km2": float(flooded_area_m2 / 1e6) if np.isfinite(flooded_area_m2) else float("nan"),
            "area_map_m2": area_map,
            "threshold": self.threshold,
            "crs": str(crs),
            "input_shape": tuple(shape),
        }

    def predict_npz(self, npz_path):
        """Inference from an existing processed chip (.npz, 6ch X + geo metadata)."""
        d = np.load(npz_path, allow_pickle=True)
        X = d["X"].astype(np.float32)
        assert X.shape == (6, 512, 512), f"expected NB02 6ch chip, got {X.shape}"
        vm = d["valid_mask"].astype(bool)
        out = self.predict(X[0], X[1], X[2], valid_mask=vm,
                           transform=np.asarray(d["transform"]), crs=str(d["crs"]))
        out["chip_id"] = str(d["chip_id"])
        out["gt"] = d["y"].astype(np.uint8)
        return out

    def save_geotiff(self, path, array, transform, crs, nodata=None, compress="deflate"):
        """Save probability (float32) or mask (uint8) preserving CRS/transform/dims."""
        import rasterio
        from rasterio.crs import CRS
        aff = transform_to_affine(transform)
        arr = np.asarray(array)
        assert arr.ndim == 2
        dtype = "float32" if arr.dtype == np.float32 else "uint8"
        profile = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
                   "count": 1, "dtype": dtype, "crs": CRS.from_string(str(crs)),
                   "transform": aff, "compress": compress}
        if nodata is not None:
            profile["nodata"] = nodata
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert not path.exists(), f"refusing to overwrite {path}"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
        return path


def visualize_inference(vv, vh, dem, probability, pred_mask, gt=None, valid_mask=None,
                        title="", area_km2=None, flood_fraction=None):
    """6-panel figure: VV | VH | DEM | GT? | Prob | Pred. Returns (fig, axes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)
    dem = np.asarray(dem, dtype=np.float32)
    prob = np.asarray(probability, dtype=np.float32)
    pred = np.asarray(pred_mask)
    ncols = 6
    fig, ax = plt.subplots(1, ncols, figsize=(4 * ncols, 4.5))
    ax[0].imshow(vv, cmap="gray"); ax[0].set_title("Sentinel-1 VV (dB)")
    ax[1].imshow(vh, cmap="gray"); ax[1].set_title("Sentinel-1 VH (dB)")
    ax[2].imshow(dem, cmap="terrain"); ax[2].set_title("DEM (m)")
    if gt is not None:
        g = np.asarray(gt).astype(float)
        if valid_mask is not None:
            disp = np.where(np.asarray(valid_mask).astype(bool), g, -1)
        else:
            disp = g
        ax[3].imshow(disp, cmap=ListedColormap(["#808080", "#1a1a1a", "#00b3ff"]),
                     vmin=-1, vmax=1, interpolation="nearest")
        ax[3].set_title("Ground Truth (grey=invalid)")
    else:
        ax[3].imshow(np.zeros_like(vv), cmap="gray")
        ax[3].set_title("Ground Truth (n/a)")
    im = ax[4].imshow(prob, cmap="Blues", vmin=0, vmax=1); ax[4].set_title("Flood Probability")
    fig.colorbar(im, ax=ax[4], fraction=0.046, pad=0.04)
    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool)
        disp_pred = np.where(vm, pred, 2)
        ax[5].imshow(disp_pred, cmap=ListedColormap(["#000000", "#ffffff", "#808080"]),
                     vmin=0, vmax=2, interpolation="nearest")
    else:
        ax[5].imshow(pred, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[5].set_title("Predicted Flood Mask")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    a_txt = f"{area_km2:.2f} km2" if area_km2 is not None and np.isfinite(area_km2) else "n/a"
    f_txt = f"{100 * flood_fraction:.2f} %" if flood_fraction is not None else "n/a"
    fig.suptitle(f"{title}\nPredicted flooded area: {a_txt} | Flood fraction: {f_txt}")
    fig.tight_layout()
    return fig, ax


def _read_input_array(path):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32), None, None
    try:
        import rasterio
        with rasterio.open(p) as src:
            return src.read(1).astype(np.float32), src.transform, src.crs
    except Exception:
        return np.load(p).astype(np.float32), None, None


def main(argv=None):
    ap = argparse.ArgumentParser(description="M1 flood inference (VV/VH/DEM, thr 0.5, locked).")
    ap.add_argument("--npz", default=None, help="processed chip .npz (uses X[0,1,2]+valid_mask+geo)")
    ap.add_argument("--vv", default=None); ap.add_argument("--vh", default=None)
    ap.add_argument("--dem", default=None, help=".tif (geo) or .npy")
    ap.add_argument("--valid-mask", default=None, help=".npy bool mask (optional)")
    ap.add_argument("--mean-lat", type=float, default=None,
                    help="mean latitude (deg) for area calc when no geotransform is available")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--prefix", default="m1_pred")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-viz", action="store_true")
    args = ap.parse_args(argv)

    pred = FloodPredictor(device=args.device)
    transform = None; crs = "EPSG:4326"; chip = args.prefix; gt = None; vm = None

    if args.npz is not None:
        out = pred.predict_npz(args.npz)
        prob, mask, vm = out["probability"], out["mask"], out["valid_mask"]
        transform = transform_to_affine(np.load(args.npz, allow_pickle=True)["transform"])
        crs = str(np.load(args.npz, allow_pickle=True)["crs"])
        chip = out["chip_id"]; gt = out["gt"]
        res = out
    else:
        assert args.vv and args.vh and args.dem, "--npz or (--vv --vh --dem) required"
        vv, t1, c1 = _read_input_array(args.vv)
        vh, t2, c2 = _read_input_array(args.vh)
        dem, t3, c3 = _read_input_array(args.dem)
        transform = t1 or t2 or t3
        crs = c1 or c2 or c3 or "EPSG:4326"
        if args.valid_mask is not None:
            vm = np.load(args.valid_mask).astype(bool)
        res = pred.predict(vv, vh, dem, valid_mask=vm, transform=transform, crs=str(crs))
        prob, mask = res["probability"], res["mask"]
        if transform is None and args.mean_lat is not None:
            H, W = prob.shape
            # degree pixel size of Sen1Floods11 chips; latitude-corrected area
            res_deg = 8.9831528e-05
            lat = np.full(H, args.mean_lat)
            faux = [res_deg, 0.0, 0.0, 0.0, -res_deg, args.mean_lat + H / 2 * res_deg]
            area_map = pixel_area_map_m2((H, W), faux)
            res["flooded_area_m2"] = float(area_map[(mask == 1) & res["valid_mask"]].sum())
            res["flooded_area_km2"] = float(res["flooded_area_m2"] / 1e6)

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    prob_path = outdir / f"{chip}_{args.prefix}_prob.tif"
    mask_path = outdir / f"{chip}_{args.prefix}_mask.tif"
    if transform is not None:
        pred.save_geotiff(prob_path, prob, transform, crs)
        pred.save_geotiff(mask_path, mask.astype(np.uint8), transform, crs)
    else:
        prob_path = outdir / f"{chip}_{args.prefix}_prob.npy"
        mask_path = outdir / f"{chip}_{args.prefix}_mask.npy"
        assert not prob_path.exists() and not mask_path.exists(), "refusing to overwrite"
        np.save(prob_path, prob); np.save(mask_path, mask)
    viz_path = None
    if args.save_viz:
        src = np.load(args.npz, allow_pickle=True) if args.npz else None
        _vv = src["X"][0] if src is not None else vv
        _vh = src["X"][1] if src is not None else vh
        _dem = src["X"][2] if src is not None else dem
        fig, _ = visualize_inference(_vv, _vh, _dem, prob, mask, gt=gt, valid_mask=vm,
                                     title=f"M1 {chip}",
                                     area_km2=res["flooded_area_km2"],
                                     flood_fraction=res["flood_fraction"])
        viz_path = outdir / f"{chip}_{args.prefix}_viz.png"
        assert not viz_path.exists(), f"refusing to overwrite {viz_path}"
        fig.savefig(viz_path, dpi=120)
        import matplotlib.pyplot as plt
        plt.close(fig)

    print(f"input dimensions: {res['input_shape']}")
    print(f"valid pixel count: {res['valid_pixel_count']}")
    print(f"flooded pixel count: {res['flooded_pixel_count']}")
    print(f"flood fraction: {res['flood_fraction']:.6f}")
    print(f"flooded area km2: {res['flooded_area_km2']:.4f}")
    print(f"probability path: {prob_path}")
    print(f"mask path: {mask_path}")
    if viz_path is not None:
        print(f"visualization path: {viz_path}")
    return {"probability_path": str(prob_path), "mask_path": str(mask_path),
            "viz_path": str(viz_path) if viz_path else None, **{k: res[k] for k in
            ("valid_pixel_count", "flooded_pixel_count", "flood_fraction",
             "flooded_area_m2", "flooded_area_km2", "input_shape")}}


if __name__ == "__main__":
    main()
