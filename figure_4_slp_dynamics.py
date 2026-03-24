from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib.pyplot as plt

from calculate_manuscript_data import FIGURES_DIR, save_figure_outputs


FIGURE_FILE = FIGURES_DIR / "figure_4_slp_dynamics.png"


def placeholder(ax, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    placeholder(
        axes[0],
        "(a) Summer SLP variance reduction",
        "Pending yearly SLP-variance data",
    )
    placeholder(
        axes[1],
        "(b) Shoulder-season poleward shift",
        "Pending yearly SLP-variance data",
    )
    local_path, overleaf_path = save_figure_outputs(fig, FIGURE_FILE.name, dpi=300)
    plt.close(fig)
    print(local_path)
    print(overleaf_path)


if __name__ == "__main__":
    main()
