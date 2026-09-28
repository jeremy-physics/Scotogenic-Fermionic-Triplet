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

from plot_template import c5, c6, c_esc1

DATA_PATH = ROOT / "outputs" / "analyses" / "analysis_dd" / "data" / "scans" / "scan_lfv.dat"
OUTPUT_PATH = Path(__file__).with_name("DD_fscoto_positive.pdf")
STYLE_PATH = ROOT / "styles" / "paper_style.mplstyle"

LZ = 7.413460268587916e-47
PURE_EW_SIGMA = 2.434770333230314e-47

FTREE_SCALE = 1e-8
YUKAWA_LIMIT = np.sqrt(4.0 * np.pi)
POINT_SIZE = 9
CROSS_SIZE = 13
X_LIMITS = (2e-5, 4e-1)
Y_LIMITS = (1.7e-47, 0.8e-44)


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

x_values = ftree / FTREE_SCALE
y_values = sigma_si
mask_yukawa = np.isfinite(yXi_max) & (yXi_max <= YUKAWA_LIMIT)
mask_base = (
    mask_yukawa
    & (ftree > 0.0)
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
fig, ax = plt.subplots(figsize=(6.0, 5.0))
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
ax.axhline(LZ, color=c_esc1, linestyle="-", linewidth=1.2, zorder=3)
ax.axhline(PURE_EW_SIGMA, color="red", linestyle="-", linewidth=1.1, zorder=3)


# === TEXTS, ANNOTATIONS AND MARKERS ===
line_label_style = {
    "va": "bottom",
    "ha": "right",
    "fontsize": 11,
    "zorder": 10,
}
branch_label_style = {
    "va": "bottom",
    "ha": "right",
    "fontsize": 14,
    "zorder": 10,
}
ax.annotate(
    r"LUX-ZEPLIN $(4.5\mathrm{t}\times \mathrm{y})$",
    xy=(9e-4, LZ),
    xytext=(0, 3),
    textcoords="offset points",
    color=c_esc1,
    **line_label_style,
)
ax.text(
    0.65,
    PURE_EW_SIGMA * 1.04,
    r"Minimal fermionic triplet",
    transform=trans_xaxes_ydata,
    color="red",
    va="bottom",
    ha="left",
    fontsize=10,
    zorder=10,
)
ax.annotate(
    r"$f_q^{\rm scoto}>0$",
    xy=(1.5e-4, 4.5e-45),
    xytext=(0, 0),
    textcoords="offset points",
    color="black",
    bbox={
        "boxstyle": "round,pad=0.3",
        "facecolor": "white",
        "edgecolor": "black",
        "alpha": 0.9,
    },
    **branch_label_style,
)


# === AXES AND LEGEND ===
ax.set_xlabel(r"$|f_q^{\rm scoto}|/(10^{-8}\,{\rm GeV}^{-3})$")
ax.set_ylabel(r"$\sigma_{\rm SI}\ [{\rm cm}^2]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*X_LIMITS)
ax.set_ylim(*Y_LIMITS)


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
