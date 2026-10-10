"""T-SAO setup verification: run from the repo root after installing requirements.

Checks (read-only, CPU):
  1. Python version + PyTorch import/device
  2. Production checkpoint / config / normalization stats exist and agree
  3. Real M1 inference on the demo scene Mekong_1443339 (expects ~18.2236 km2)
  4. Demo scene files for the website (4 test chips)

Usage (Windows PowerShell, from repo root):
  python scripts/verify_setup.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED_AREA = {"Mekong_1443339": 18.2236, "Ghana_1078550": 0.0097}
SCENES = ["Mekong_1443339", "Ghana_1078550", "Spain_7387658", "India_44475"]

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main():
    print(f"repo root: {ROOT}")
    print(f"python: {sys.version.split()[0]}")
    check("python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])

    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        check("torch import", True, f"{torch.__version__} device={dev}")
    except ImportError as e:
        check("torch import", False, str(e))
        print(f"\n{len(failures)} check(s) FAILED.")
        sys.exit(1)

    ckpt = ROOT / "models" / "nb06_m1_s1_dem_best.pt"
    cfg_p = ROOT / "models" / "final_model_config.json"
    stats_p = ROOT / "processed" / "normalization_stats.json"
    check("checkpoint exists", ckpt.exists(), str(ckpt))
    check("model config exists", cfg_p.exists(), str(cfg_p))
    check("normalization stats exist", stats_p.exists(), str(stats_p))
    if failures:
        print(f"\n{len(failures)} check(s) FAILED.")
        sys.exit(1)

    cfg = json.loads(cfg_p.read_text())
    check("config is M1/thr-0.5", cfg.get("model_name") == "M1" and cfg.get("threshold") == 0.5,
          f"{cfg.get('model_name')} thr={cfg.get('threshold')}")

    from src.inference import FloodPredictor

    try:
        pred = FloodPredictor(device="cpu")
        check("FloodPredictor loads (params/thr/stats asserts)",
              True, f"thr={pred.threshold} mean={pred.mean.ravel().tolist()}")
    except Exception as e:
        check("FloodPredictor loads (params/thr/stats asserts)", False, str(e))
        print(f"\n{len(failures)} check(s) FAILED.")
        sys.exit(1)

    for scene, want in EXPECTED_AREA.items():
        p = ROOT / "processed" / "test" / f"{scene}.npz"
        if not p.exists():
            check(f"inference {scene}", False, f"missing {p}")
            continue
        out = pred.predict_npz(str(p))
        got = round(float(out["flooded_area_km2"]), 4)
        check(f"inference {scene}", abs(got - want) < 0.01,
              f"area={got} km2 (locked {want}), flooded_px={out['flooded_pixel_count']}")

    missing = [s for s in SCENES if not (ROOT / "processed" / "test" / f"{s}.npz").exists()]
    check("website demo scenes present", not missing,
          "all 4 present" if not missing else f"missing: {missing}")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    print("All checks passed. Start the website per README section 'Run T-SAO Locally'.")


if __name__ == "__main__":
    main()
