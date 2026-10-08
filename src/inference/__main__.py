"""CLI entry: python -m src.inference --npz <chip> --output <dir> [--save-viz]."""
from .flood_predictor import main

if __name__ == "__main__":
    main()
