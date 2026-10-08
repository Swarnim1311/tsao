"""Server-side raster rendering for the T-SAO demo (PIL only, no matplotlib).

All layers are rendered from REAL pipeline arrays. Palette follows the site
design system: deep charcoal base, teal flood accent, muted terrain ramp.
"""
import io
import numpy as np
from PIL import Image

SIZE = 512

# --- design-system palette -------------------------------------------------
LAND = (16, 20, 19)        # #101413 near-black green-charcoal
FLOOD = (64, 186, 164)     # #40baa4 sophisticated teal accent
NODATA = (52, 58, 56)      # muted grey for invalid pixels
NONFLOOD_DIM = (26, 33, 31)


def _stretch(a, lo=2, hi=98, mask=None):
    a = np.asarray(a, dtype=np.float64)
    ref = a[mask] if mask is not None and np.any(mask) else a[np.isfinite(a)]
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.zeros_like(a, dtype=np.float64)
    v0, v1 = np.percentile(ref, [lo, hi])
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
        return np.zeros_like(a, dtype=np.float64)
    return np.clip((a - v0) / (v1 - v0), 0, 1)


def gray_layer(a, valid):
    g = (_stretch(a, mask=valid) * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[~valid] = NODATA
    return Image.fromarray(rgb)


def dem_layer(a, valid):
    t = _stretch(a, mask=valid)
    # muted terrain ramp: deep pine -> olive -> sand -> pale rock
    stops = np.array([
        [24, 46, 38], [74, 96, 62], [139, 133, 96], [196, 188, 160], [226, 222, 208],
    ], dtype=np.float64) / 255.0
    pos = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(t, pos, stops[:, i]) for i in range(3)], axis=-1)
    rgb[~valid] = np.array(NODATA) / 255.0
    return Image.fromarray((rgb * 255).astype(np.uint8))


def prob_layer(p, valid):
    p = np.clip(np.asarray(p, dtype=np.float64), 0, 1)
    base = np.array(LAND, dtype=np.float64) / 255.0
    acc = np.array(FLOOD, dtype=np.float64) / 255.0
    rgb = base[None, None, :] * (1 - p[..., None]) + acc[None, None, :] * p[..., None]
    rgb[~valid] = np.array(NODATA) / 255.0
    return Image.fromarray((rgb * 255).astype(np.uint8))


def mask_layer(m, valid):
    m = np.asarray(m).astype(bool)
    v = np.asarray(valid).astype(bool)
    rgb = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    rgb[:, :] = NONFLOOD_DIM
    rgb[m & v] = FLOOD
    rgb[~v] = NODATA
    return Image.fromarray(rgb)


def gt_layer(g, valid):
    return mask_layer(np.asarray(g).astype(bool), valid)


def to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
