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

from plot_template import c0, c1, c5, c6

DATA_PATH = ROOT / "outputs" / "analyses" / "analysis_dd" / "data" / "scans" / "scan_lfv.dat"
OUTPUT_PATH = Path(__file__).with_name("fig02_interference_ratio.pdf")
STYLE_PATH = ROOT / "styles" / "paper_style_colorbar.mplstyle"

M_N = 0.939
CFACT = 0.389379e-27
LZ = 7.413460268587916e-47
XENONNT = 4.0920440344790904e-47
DARKSIDE = 1.9212420771126029e-47
DARWIN = 5.895523679499314e-48
ARGO = 3.6753483466448516e-48
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
mXi = scan["mXi"]
sigma_si = scan["sigma_si"]
fEW = scan["fEW"]
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

with np.errstate(divide="ignore", invalid="ignore"):
    rN = ftree / fEW
    sigma_ew = 4.0 / np.pi * ((mXi * M_N) / (mXi + M_N)) ** 2 * fEW**2 * CFACT
    sigma_ratio = sigma_si / sigma_ew

mask_yukawa = np.isfinite(yXi_max) & (yXi_max <= YUKAWA_LIMIT)
mask_base = (
    mask_yukawa
    & np.isfinite(rN)
    & np.isfinite(sigma_ratio)
    & (sigma_ratio > 0.0)
)
mask_good = mask_base & (np.cos(alpha) > 0.95)
mask_out = mask_base & ~mask_good

x_line = np.linspace(-5.0, 5.0, 4000)
y_line = np.abs(1.0 + x_line) ** 2


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
    rN[mask_out],
    sigma_ratio[mask_out],
    marker="x",
    color=c5,
    linewidths=0.6,
    s=CROSS_SIZE,
    alpha=0.75,
    rasterized=True,
    zorder=1,
)
ax.scatter(
    rN[mask_good],
    sigma_ratio[mask_good],
    marker="o",
    color=c6,
    edgecolors="none",
    linewidths=0.0,
    s=POINT_SIZE,
    alpha=0.9,
    rasterized=True,
    zorder=2,
)
ax.plot(x_line, y_line, color=c6, linewidth=1.0, zorder=3)


# === LINES AND REGIONS ===
ax.axvline(-1.0, color=c1, linestyle="--", linewidth=1.2, zorder=4)
ax.axhline(1.0, color=c0, linestyle="--", linewidth=1.0, zorder=2)


# === TEXTS, ANNOTATIONS AND MARKERS ===
ax.text(
    0.39,
    2.0e1,
    r"$r_N=-1$",
    transform=trans_xaxes_ydata,
    va="center",
    ha="left",
    fontsize=14,
    color=c1,
    rotation=90,
    zorder=10,
)
ax.text(
    0.64,
    2.1,
    r"Constructive interference",
    transform=trans_xaxes_ydata,
    va="center",
    ha="left",
    fontsize=10,
    color=c0,
    zorder=10,
)
ax.text(
    0.64,
    0.35,
    r"Destructive interference",
    transform=trans_xaxes_ydata,
    va="center",
    ha="left",
    fontsize=10,
    color=c0,
    zorder=10,
)


# === AXES AND LEGEND ===
ax.set_xlabel(r"$r_N$")
ax.set_ylabel(r"$\sigma^N_{\rm SI}/\sigma^{N,\rm EW}_{\rm SI}$")
ax.set_yscale("log")
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(1e-5, 8e1)


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
