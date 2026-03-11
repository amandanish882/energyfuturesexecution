"""Plot styling utilities for consistent visualization across the project.

Applies a shared matplotlib style (seaborn whitegrid) and standardised
rcParams to every figure produced by the platform. Also exposes a
canonical colour palette keyed by commodity ticker and semantic role.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

_STYLE_APPLIED = False


def apply_style():
    """Apply the project-wide matplotlib style and rcParams settings.

    Activates the ``seaborn-v0_8-whitegrid`` base style and overrides
    a set of rcParams to enforce consistent figure dimensions, font
    sizes, line widths, and DPI across all plots in the project. The
    function is idempotent: subsequent calls after the first are no-ops,
    guarded by the module-level ``_STYLE_APPLIED`` flag.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "figure.figsize": (12, 6),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 1.5,
        "figure.dpi": 100,
    })
    _STYLE_APPLIED = True


COLOURS = {
    "CL": "#1f77b4",
    "HO": "#ff7f0e",
    "RB": "#2ca02c",
    "NG": "#d62728",
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "warning": "#d62728",
    "grid": "#cccccc",
}
