"""Shared matplotlib style for publication-quality figures (Computer Modern via LaTeX)."""
import matplotlib.pyplot as plt


def apply_style():
    """Apply LaTeX Computer Modern style to all matplotlib plots."""
    plt.rcParams.update({
        # LaTeX rendering
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}",
        "font.family": "serif",

        # Font sizes (+1 from base)
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,

        # Grid
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,

        # Axes
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,

        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",

        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
