from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib import transforms


# === GLOBAL PLOT CONFIGURATION ===
ROOT = Path(__file__).resolve().parents[4]
PLOT_TEMPLATE_PATH = ROOT / "scripts"
if str(PLOT_TEMPLATE_PATH) not in sys.path:
    sys.path.insert(0, str(PLOT_TEMPLATE_PATH))

from plot_template import (
    c0,
    c1,
    c2,
    c4,
    c5,
    c6,
)

DATA_PATH = ROOT / "outputs" / "analyses" / "analysis_dd" / "data" / "scans" / "scan_lfv.dat"
OUTPUT_PATH = Path(__file__).with_name("fig08_sigma_alpha.pdf")
STYLE_PATH = ROOT / "styles" / "paper_style_colorbar.mplstyle"

LZ = 8.78658722623048e-47
XENONNT = 4.484204325419306e-47
DARWIN = 2.67279895071297e-47
ARGO = 8.130408367888161e-48
PURE_EW_SIGMA = 2.15e-47

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
sigma_si = scan["sigma_si"]
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

x_values = np.cos(alpha)
y_values = sigma_si
mask_yukawa = np.isfinite(yXi_max) & (yXi_max <= YUKAWA_LIMIT)
mask_base = mask_yukawa & np.isfinite(x_values) & np.isfinite(y_values) & (y_values > 0.0)
mask_good = mask_base & (np.cos(alpha) > 0.95)
mask_out = mask_base & ~mask_good
x_band = np.linspace(-1.6, 1.6, 200)


# === STYLE ===
plt.style.use(STYLE_PATH)
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r""
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["pgf.rcfonts"] = False
plt.rc("font", family="serif")


# === FIGURE AND AXES ===
fig, ax = plt.subplots(figsize=(7.0, 5.0))
trans_xaxes_ydata = transforms.blended_transform_factory(ax.transAxes, ax.transData)


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
ax.axhline(LZ, color=c1, linestyle="-", linewidth=1.2, label=r"LZ", zorder=3)
ax.axhline(PURE_EW_SIGMA, color=c2, linestyle=":", linewidth=1.3, label=r"Pure EW", zorder=3)
ax.axhline(XENONNT, color=c5, linestyle=":", linewidth=1.3, label=r"XENONnT", zorder=3)
ax.axhline(DARWIN, color=c4, linestyle=":", linewidth=1.3, label=r"DARWIN", zorder=3)
ax.axhline(ARGO, color=c0, linestyle=":", linewidth=1.3, label=r"ARGO", zorder=3)
ax.fill_between(x_band, LZ, 1e-40, color=c1, alpha=0.28, zorder=0, rasterized=True)


# === TEXTS, ANNOTATIONS AND MARKERS ===
ax.text(
    0.04,
    9e-46,
    r"Excluded by LZ",
    transform=trans_xaxes_ydata,
    va="center",
    ha="left",
    fontsize=16,
    color=c1,
    zorder=10,
)


# === AXES AND LEGEND ===
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\sigma^N_{\rm SI}\ [{\rm cm}^2]$")
ax.set_yscale("log")
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(8e-51, 2e-45)
legend = ax.legend(fontsize=10, loc="lower left", frameon=True)
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_boxstyle("Round,pad=0.1")


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
