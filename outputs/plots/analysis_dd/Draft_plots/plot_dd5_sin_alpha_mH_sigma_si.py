from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib import colors


# === GLOBAL PLOT CONFIGURATION ===
ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = ROOT / "src"
PLOT_TEMPLATE_PATH = ROOT / "scripts"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PLOT_TEMPLATE_PATH) not in sys.path:
    sys.path.insert(0, str(PLOT_TEMPLATE_PATH))

from targets import CHI_TARGETS
from variables import get_variable_label
from plot_template import c6

DATA_PATH = ROOT / "outputs" / "analyses" / "analysis_dd" / "data" / "scans" / "scan_lfv.dat"
OUTPUT_PATH = Path(__file__).with_name("DD_sin_alpha_mH_sigma_si.pdf")
STYLE_PATH = ROOT / "styles" / "paper_style_colorbar.mplstyle"

YUKAWA_LIMIT = np.sqrt(4.0 * np.pi)
POINT_SIZE = 9
POINT_EDGE_COLOR = "none"
POINT_LINEWIDTH = 0.0
X_LIMITS = (2.0e2, 1.0e6)
Y_LIMITS = (-0.55, 0.55)
SIGMA_REFERENCE = 2.06e-47
SIGMA_REFERENCE_ERROR = 0.24e-47
SIGMA_GREEN_LOW = SIGMA_REFERENCE - SIGMA_REFERENCE_ERROR
SIGMA_GREEN_HIGH = SIGMA_REFERENCE + SIGMA_REFERENCE_ERROR
SIGMA_COLOR_MIN = 1.0e-55
SIGMA_COLOR_MAX = 1.0e-44
BLUE_COLOR_BINS = 24
GREEN_COLOR_BINS = 4
RED_COLOR_BINS = 12


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
mHH = scan["mHH"]
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

x_values = mHH
y_values = np.sin(alpha)
kappa_f = np.cos(alpha)
kappa_target = CHI_TARGETS["kappaF"]
kappa_pull = np.abs((kappa_f - kappa_target["exp"]) / kappa_target["err"])

mask_yukawa = np.isfinite(yXi_max) & (yXi_max <= YUKAWA_LIMIT)
mask_base = (
    mask_yukawa
    & np.isfinite(x_values)
    & np.isfinite(y_values)
    & np.isfinite(sigma_si)
    & np.isfinite(kappa_pull)
    & (x_values > 0.0)
    & (sigma_si > 0.0)
)
mask_plot = mask_base & (kappa_pull <= 3.0)

with np.errstate(divide="ignore", invalid="ignore"):
    log10_sigma_si = np.log10(sigma_si)

blue_boundaries = np.linspace(
    np.log10(SIGMA_COLOR_MIN),
    np.log10(SIGMA_GREEN_LOW),
    BLUE_COLOR_BINS + 1,
)
green_boundaries = np.linspace(
    np.log10(SIGMA_GREEN_LOW),
    np.log10(SIGMA_GREEN_HIGH),
    GREEN_COLOR_BINS + 1,
)
red_boundaries = np.linspace(
    np.log10(SIGMA_GREEN_HIGH),
    np.log10(SIGMA_COLOR_MAX),
    RED_COLOR_BINS + 1,
)
sigma_boundaries = np.concatenate(
    [blue_boundaries, green_boundaries[1:], red_boundaries[1:]]
)


# === STYLE ===
plt.style.use(STYLE_PATH)
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r""
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["pgf.rcfonts"] = False
plt.rc("font", family="serif")


# === FIGURE AND AXES ===
fig, ax = plt.subplots(figsize=(8.0, 5.0))


# === PLOTS ===
blue_cmap = plt.get_cmap("Blues")
red_cmap = plt.get_cmap("Reds")
blue_colors = blue_cmap(np.linspace(0.35, 0.9, BLUE_COLOR_BINS))
green_colors = np.tile(np.asarray([*c6, 1.0]), (GREEN_COLOR_BINS, 1))
red_colors = red_cmap(np.linspace(0.35, 0.9, RED_COLOR_BINS))
cmap = colors.ListedColormap(
    np.vstack([blue_colors, green_colors, red_colors]),
    name="sigma_blue_green_red",
)
norm = colors.BoundaryNorm(sigma_boundaries, cmap.N, clip=True)

scatter = ax.scatter(
    x_values[mask_plot],
    y_values[mask_plot],
    c=log10_sigma_si[mask_plot],
    cmap=cmap,
    norm=norm,
    marker="o",
    edgecolors=POINT_EDGE_COLOR,
    linewidths=POINT_LINEWIDTH,
    s=POINT_SIZE,
    alpha=0.9,
    rasterized=True,
    zorder=2,
)


# === LINES AND REGIONS ===


# === TEXTS, ANNOTATIONS AND MARKERS ===


# === AXES AND LEGEND ===
ax.set_xlabel(get_variable_label("mHH") + r"$\ [{\rm GeV}]$")
ax.set_ylabel(r"$\sin\alpha$")
ax.set_xscale("log")
ax.set_xlim(*X_LIMITS)
ax.set_ylim(*Y_LIMITS)

cbar_ax = ax.inset_axes([0.0, 1.03, 1.0, 0.045], transform=ax.transAxes)
cbar = fig.colorbar(
    scatter,
    cax=cbar_ax,
    orientation="horizontal",
    boundaries=sigma_boundaries,
    spacing="uniform",
)
cbar.set_label(get_variable_label("sigma_si") + r"$\ [{\rm cm}^2]$", fontsize=16)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.set_ticks(
    [
        -54.0,
        -51.0,
        -48.0,
        np.log10(SIGMA_REFERENCE),
        -46.0,
        -45.0,
    ]
)
cbar.ax.set_xticklabels(
    [
        r"$10^{-54}$",
        r"$10^{-51}$",
        r"$10^{-48}$",
        r"$2.06\times10^{-47}$",
        r"$10^{-46}$",
        r"$10^{-45}$",
    ]
)
cbar.ax.tick_params(direction="in", labelsize=9, pad=2)


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Points within 3 sigma: {np.count_nonzero(mask_plot)}")
print(f"Saved {OUTPUT_PATH}")
