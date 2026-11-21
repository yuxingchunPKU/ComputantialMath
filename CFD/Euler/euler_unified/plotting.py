"""Matplotlib output for the unified Euler cases."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import output_dir

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
COLOR_NUM = "#2a78d6"
COLOR_EXACT = "#eb6834"

QUANTITIES = (
    ("rho", "Density $\\rho$", "rho"),
    ("u", "Velocity $u$", "u"),
    ("p", "Pressure $p$", "p"),
)


def _style_axis(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=10, pad=8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, width=0.7, length=3)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _plot_quantity(ax, result, key, label, cfg, title):
    x = result["x"]
    ax.plot(x, result[key], color=COLOR_NUM, linewidth=2.0,
            label="Numerical")
    exact = result.get(f"{key}_exact")
    if exact is not None:
        ax.plot(x, exact, color=COLOR_EXACT, linewidth=1.8,
                linestyle="--", label="Exact")
    # Nearly constant quantities (e.g. u and p in the smooth advection case)
    # would otherwise autoscale to a 1e-15 window with offset tick labels.
    values = [np.asarray(result[key], dtype=float)]
    if exact is not None:
        values.append(np.asarray(exact, dtype=float))
    lo = min(float(np.min(v)) for v in values)
    hi = max(float(np.max(v)) for v in values)
    scale = max(abs(lo), abs(hi), 1.0)
    if hi - lo < 1.0e-8 * scale:
        center = 0.5 * (lo + hi)
        pad = max(0.1 * abs(center), 1.0e-3)
        ax.set_ylim(center - pad, center + pad)
        ax.ticklabel_format(axis="y", useOffset=False)
    _style_axis(ax, "$x$", label, title)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY,
              loc="best")


def plot_case(cfg: dict, result: dict) -> dict:
    """Save individual and combined figures; return their paths."""
    out = output_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    dpi = cfg["output"]["dpi"]
    figsize = tuple(cfg["output"]["figsize"])
    name = cfg["name"]
    paths = {}

    for key, label, _ in QUANTITIES:
        fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
        _plot_quantity(ax, result, key, label, cfg,
                       f"{name}: {label}")
        fig.tight_layout()
        path = out / f"{key}.png"
        fig.savefig(path, dpi=dpi, facecolor=SURFACE)
        plt.close(fig)
        paths[key] = path

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1), facecolor=SURFACE)
    for ax, (key, label, _) in zip(axes, QUANTITIES):
        _plot_quantity(ax, result, key, label, cfg, label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9,
               labelcolor=TEXT_SECONDARY, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out / "overview.png"
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    paths["overview"] = path
    return paths
