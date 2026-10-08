"""Production M1 inference package (lazy imports keep `python -m` clean)."""
_LAZY = {
    "FloodPredictor": ".flood_predictor",
    "compute_metrics": ".flood_predictor",
    "pixel_area_map_m2": ".flood_predictor",
    "visualize_inference": ".flood_predictor",
    "THRESHOLD": ".flood_predictor",
    "INPUT_CHANNEL_NAMES": ".flood_predictor",
    "INPUT_CHANNEL_IDX": ".flood_predictor",
    "MODEL_NAME": ".flood_predictor",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
