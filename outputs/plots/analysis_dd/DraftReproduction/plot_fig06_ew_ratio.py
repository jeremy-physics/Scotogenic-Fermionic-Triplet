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
OUTPUT_PATH = Path(__file__).with_name("fig06_ew_ratio.pdf")
STYLE_PATH = ROOT / "styles" / "paper_style_colorbar.mplstyle"

SM_EW_AMPLITUDE = 2.2198e-10
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
mh = scan["mh"]
mHH = scan["mHH"]
fEW = scan["fEW"]
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

with np.errstate(divide="ignore", invalid="ignore"):
    x_values = np.sin(alpha) ** 2 * (1.0 - mh**2 / mHH**2)
    y_values = fEW / SM_EW_AMPLITUDE

mask_yukawa = np.isfinite(yXi_max) & (yXi_max <= YUKAWA_LIMIT)
mask_base = mask_yukawa & np.isfinite(x_values) & np.isfinite(y_values)
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
ax.set_xlabel(r"$\sin^2(\alpha)(1-m_h^2/m_H^2)$")
ax.set_ylabel(r"$\mathcal{A}_{N}^{\rm EW}/\mathcal{A}_{N}^{\rm EW,SM}$")
ax.set_yscale("log")
ax.set_xlim(-0.01, 0.22)
ax.set_ylim(0.95, 5)
#ax.set_xlim(-0.01, 0.12)
#ax.set_ylim(0.95, 1.35)


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
