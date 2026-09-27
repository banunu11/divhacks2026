"""Shared chart style for the notebooks.

    from ml.plotting import setup, SERIES, GRAY, SEQ_CMAP, DIV_CMAP, show
    setup()

Colors are a colorblind-checked palette: use SERIES in order (never cycle
past it), SEQ_CMAP for magnitude, DIV_CMAP for +/- around zero.
"""
import sys

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from ml import config

FIGURES_DIR = config.ROOT / "reports" / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
GRAY = "#b5b3ad"  # "everything else" when one series is highlighted

# categorical slots, in fixed order: blue, orange, aqua, yellow, magenta, green, violet, red
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8d8c", "#b3261e"])


def _interactive() -> bool:
    try:
        return get_ipython() is not None  # noqa: F821  (defined inside IPython/Jupyter)
    except NameError:
        return False


def setup() -> None:
    if not _interactive():
        mpl.use("Agg")  # running as a plain script: save PNGs instead of opening windows
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "figure.figsize": (9, 4.2),
        "font.size": 10,
        "text.color": TEXT,
        "axes.labelcolor": TEXT_2,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def show(fig, name: str) -> None:
    """Display inline in a notebook, or save to reports/figures/<name>.png."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if _interactive():
        plt.show()
    else:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"  saved {path.relative_to(config.ROOT)}")
    plt.close(fig)
