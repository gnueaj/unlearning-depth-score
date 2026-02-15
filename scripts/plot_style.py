"""Shared matplotlib style for publication-quality figures (Roboto font)."""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _register_roboto():
    """Register Roboto font from font-roboto package."""
    try:
        import font_roboto, os
        font_dir = os.path.join(os.path.dirname(font_roboto.__file__), "files")
        for f in os.listdir(font_dir):
            if f.endswith(".ttf"):
                fm.fontManager.addfont(os.path.join(font_dir, f))
    except ImportError:
        pass  # fall back to system fonts


_register_roboto()


def apply_style():
    """Apply Roboto font style to all matplotlib plots."""
    plt.rcParams.update({
        # Font (Roboto, no LaTeX)
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto", "DejaVu Sans", "Arial"],
        "mathtext.fontset": "dejavusans",

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
