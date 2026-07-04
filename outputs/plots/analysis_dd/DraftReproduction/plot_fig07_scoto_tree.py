from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt


# === GLOBAL PLOT CONFIGURATION ===
ROOT = Path(__file__).resolve().parents[4]
PLOT_TEMPLATE_PATH = ROOT / "scripts"
if str(PLOT_TEMPLATE_PATH) not in sys.path:
    sys.path.insert(0, str(PLOT_TEMPLATE_PATH))

from plot_template import c5, c6

DATA_PATH = ROOT / "outputs" / "analyses" / "analysis_dd" / "data" / "scans" / "scan_lfv.dat"
OUTPUT_PATH = Path(__file__).with_name("fig07_scoto_tree.pdf")
STYLE_PATH = ROOT / "styles" / "paper_style_colorbar.mplstyle"

YUKAWA_LIMIT = np.sqrt(4.0 * np.pi)
POINT_SIZE = 9
CROSS_SIZE = 13


# === DATA LOADING AND PREPARATION ===
def load_scan(path):
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().lstrip("#").strip()
    names = [name.strip() for name in header.split(",")]
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {name: data[:, index] for index, name in enumerate(names)}


scan = load_scan(DATA_PATH)
alpha = scan["alpha"]
ftree = scan["ftree"]
fscoto = scan["fscoto"]
yXi_max = np.maximum.reduce(
    [
        scan["yXi11_abs"],
        scan["yXi12_abs"],
        scan["yXi21_abs"],
        scan["yXi22_abs"],
        scan["yXi31_abs"],
        scan["yXi32_abs"],
    ]
)

x_values = np.abs(ftree)
y_values = np.abs(fscoto)
mask_yukawa = np.isfinite(yXi_max) & (yXi_max <= YUKAWA_LIMIT)
mask_base = (
    mask_yukawa
    & np.isfinite(x_values)
    & np.isfinite(y_values)
    & (x_values > 0.0)
    & (y_values > 0.0)
)
mask_good = mask_base & (np.cos(alpha) > 0.95)
mask_out = mask_base & ~mask_good


# === STYLE ===
plt.style.use(STYLE_PATH)
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r""
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["pgf.rcfonts"] = False
plt.rc("font", family="serif")


# === FIGURE AND AXES ===
fig, ax = plt.subplots(figsize=(7.0, 5.0))


# === PLOTS ===
ax.scatter(
    x_values[mask_out],
    y_values[mask_out],
    marker="x",
    color=c5,
    linewidths=0.6,
    s=CROSS_SIZE,
    alpha=0.75,
    rasterized=True,
    zorder=1,
)
ax.scatter(
    x_values[mask_good],
    y_values[mask_good],
    marker="o",
    color=c6,
    edgecolors="none",
    linewidths=0.0,
    s=POINT_SIZE,
    alpha=0.9,
    rasterized=True,
    zorder=2,
)


# === LINES AND REGIONS ===


# === TEXTS, ANNOTATIONS AND MARKERS ===


# === AXES AND LEGEND ===
ax.set_xlabel(r"$|f^{\rm tree}_q|$")
ax.set_ylabel(r"$|f^{\rm scot}_q|$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(5e-18, 2e-8)
ax.set_ylim(5e-24, 2e-15)


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
