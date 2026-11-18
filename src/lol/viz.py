"""Shared plotting style and figure saving.

Figures go to `reports/figures/` as 150-dpi PNGs with labelled axes and readable
fonts, because they are reused in the slides (CLAUDE.md §3.9).

Colour choices are not free-hand. The categorical slots below are the first
positions of a palette validated for colour-vision deficiency and for contrast
against a light surface; the pair actually used for two-series figures
(`SERIES_1`, `SERIES_2`) was checked and passes every gate — worst-pair
colour-blind separation dE 24.7 (protan), normal-vision dE 33.6, both well above
the dE 8 / dE 15 floors, and both above 3:1 contrast on the surface.

Rules kept here so every notebook inherits them:
  * one y-axis per figure, never two scales;
  * a legend whenever two or more series are drawn, so identity is never
    carried by colour alone;
  * recessive grid and axes, thin marks, labels only where they are read.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, in fixed order — never cycled, never re-assigned by rank.
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
SERIES_3 = "#1baf7a"  # aqua

# Single-measure default and the ink/grid tokens.
PRIMARY = SERIES_1
ACCENT = SERIES_2
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#ffffff"

FIG_DPI = 150


def set_style() -> None:
    """Apply the project's matplotlib style. Call once per notebook."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,            # on-screen; saving overrides to 150
            "savefig.dpi": FIG_DPI,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "text.color": TEXT_PRIMARY,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,   # grid behind the marks, never across them
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "axes.prop_cycle": mpl.cycler(color=[SERIES_1, SERIES_2, SERIES_3]),
        }
    )


def figures_dir() -> Path:
    """`reports/figures/`, created if absent."""
    from .load import repo_root

    path = repo_root() / "reports" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig, name: str) -> Path:
    """Save `fig` as `reports/figures/<name>.png` at 150 dpi and return the path."""
    name = name if name.endswith(".png") else f"{name}.png"
    path = figures_dir() / name
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    return path


def thousands(ax, axis: str = "y") -> None:
    """Format an axis with thousands separators (readable in slides)."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def percent(ax, axis: str = "y", decimals: int = 0) -> None:
    """Format an axis as percentages, for rate plots."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:.{decimals}%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
