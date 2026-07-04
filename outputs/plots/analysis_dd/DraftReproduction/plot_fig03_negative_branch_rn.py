from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter


# === GLOBAL PLOT CONFIGURATION ===
ROOT = Path(__file__).resolve().parents[4]
PLOT_TEMPLATE_PATH = ROOT / "scripts"
if str(PLOT_TEMPLATE_PATH) not in sys.path:
    sys.path.insert(0, str(PLOT_TEMPLATE_PATH))

from plot_template import c1, c5, c6

DATA_PATH = ROOT / "outputs" / "analyses" / "analysis_dd" / "data" / "scans" / "scan_lfv.dat"
OUTPUT_PATH = Path(__file__).with_name("fig03_negative_branch_rn.pdf")
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
mh = scan["mh"]
mHH = scan["mHH"]
vsigma = scan["vsigma"]
zXi = scan["zXi"]
fEW = scan["fEW"]
ftree = scan["ftree"]
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
    zeta_dd = zXi * np.sin(2.0 * alpha) * (1.0 - mh**2 / mHH**2)
    rN = ftree / fEW

branch = zeta_dd < 0.0
x_values = -zeta_dd
y_values = rN
mask_yukawa = np.isfinite(yXi_max) & (yXi_max <= YUKAWA_LIMIT)
mask_base = (
    mask_yukawa
    & branch
    & np.isfinite(x_values)
    & np.isfinite(y_values)
    & (x_values > 0.0)
)
mask_good = mask_base & (np.cos(alpha) > 0.95)
mask_out = mask_base & ~mask_good


def zeta_tick_label(value, _position):
    if np.isclose(value, 2e-3):
        return r"$2\times10^{-3}$"
    if np.isclose(value, 1e-2):
        return r"$10^{-2}$"
    if np.isclose(value, 3e-2):
        return r"$3\times10^{-2}$"
    return ""


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
ax.axhline(-1.0, color=c1, linestyle="--", linewidth=1.2, zorder=3)


# === TEXTS, ANNOTATIONS AND MARKERS ===
ax.text(
    0.74,
    -0.82,
    r"$r_N=-1$",
    transform=trans_xaxes_ydata,
    va="center",
    ha="left",
    fontsize=14,
    color=c1,
    zorder=10,
)


# === AXES AND LEGEND ===
ax.set_xlabel(r"$\zeta_\Xi^{\rm DD}<0$")
ax.set_ylabel(r"$r_N$")
ax.set_xscale("log")
ax.set_xlim(2e-3, 3e-2)
ax.set_ylim(-5.0, 0.0)
ax.xaxis.set_major_locator(FixedLocator([2e-3, 1e-2, 3e-2]))
ax.xaxis.set_major_formatter(FuncFormatter(zeta_tick_label))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax.xaxis.set_minor_formatter(NullFormatter())


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
