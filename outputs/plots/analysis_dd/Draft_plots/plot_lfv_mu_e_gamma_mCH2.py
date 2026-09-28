from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt


# === GLOBAL PLOT CONFIGURATION ===
ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = ROOT / "src"
PLOT_TEMPLATE_PATH = ROOT / "scripts"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PLOT_TEMPLATE_PATH) not in sys.path:
    sys.path.insert(0, str(PLOT_TEMPLATE_PATH))

from plot_template import c1, c2, c_reg1, c_esc1
from constants import BR_mu_egamma_lim, BR_mu_egamma_proy
from variables import get_variable_label

DATA_PATH = ROOT / "outputs" / "analyses" / "analysis_dd" / "data" / "scans" / "scan_lfv.dat"
OUTPUT_PATH = Path(__file__).with_name("lfv_mu_e_gamma_mCH2.pdf")
STYLE_PATH = ROOT / "styles" / "paper_style.mplstyle"

MASS_VARIABLE = "mCH2"
LZ = 7.413460268587916e-47
YUKAWA_LIMIT = np.sqrt(4.0 * np.pi)
POINT_SIZE = 9
POINT_EDGE_COLOR = "none"
POINT_LINEWIDTH = 0.0
X_LIMITS = (2.0e3, 2.0e7)
Y_LIMITS = (1.0e-31, 1.0e-8)
EXCLUDED_LABEL_X = 1.8e7
EXCLUDED_LABEL_Y = 2.0e-10
PROSPECT_LABEL_X = 1.8e7
PROSPECT_LABEL_Y = BR_mu_egamma_proy * 0.60


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
x_values = scan[MASS_VARIABLE]
y_values = scan["BR_mu_e"]
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

mask_yukawa = np.isfinite(yXi_max) & (yXi_max <= YUKAWA_LIMIT)
mask_base = (
    mask_yukawa
    & np.isfinite(x_values)
    & np.isfinite(y_values)
    & np.isfinite(sigma_si)
    & (x_values > 0.0)
    & (y_values > 0.0)
    & (sigma_si > 0.0)
)
mask_lz = mask_base & (sigma_si > LZ)
mask_kept = mask_base & ~mask_lz


# === STYLE ===
plt.style.use(STYLE_PATH)
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r""
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["pgf.rcfonts"] = False
plt.rc("font", family="serif")


# === FIGURE AND AXES ===
fig, ax = plt.subplots(figsize=(6.0, 5.0))


# === PLOTS ===
ax.scatter(
    x_values[mask_lz],
    y_values[mask_lz],
    marker="o",
    color="lightgrey",
    edgecolors=POINT_EDGE_COLOR,
    linewidths=POINT_LINEWIDTH,
    s=POINT_SIZE,
    alpha=0.6,
    rasterized=True,
    zorder=1,
)
ax.scatter(
    x_values[mask_kept],
    y_values[mask_kept],
    marker="o",
    color=c2,
    edgecolors=POINT_EDGE_COLOR,
    linewidths=POINT_LINEWIDTH,
    s=POINT_SIZE,
    alpha=0.9,
    rasterized=True,
    zorder=2,
)


# === LINES AND REGIONS ===
ax.axhspan(BR_mu_egamma_lim, Y_LIMITS[1], color=c_reg1, alpha=0.18, zorder=0)
ax.axhline(BR_mu_egamma_lim, color=c_esc1, linestyle="-", linewidth=1.2, zorder=3)
ax.axhline(
    BR_mu_egamma_proy,
    color=c1,
    linestyle="-.",
    linewidth=1.5,
    zorder=3,
)


# === TEXTS, ANNOTATIONS AND MARKERS ===
ax.text(
    EXCLUDED_LABEL_X,
    EXCLUDED_LABEL_Y,
    r"excluded by MEG",
    color=c_esc1,
    ha="right",
    va="center",
    fontsize=13,
    zorder=10,
)
ax.text(
    PROSPECT_LABEL_X,
    PROSPECT_LABEL_Y,
    r"MEG-II prospect",
    color=c1,
    ha="right",
    va="top",
    fontsize=13,
    zorder=10,
)


# === AXES AND LEGEND ===
mass_label = get_variable_label(MASS_VARIABLE).strip("$")
ax.set_xlabel(rf"${mass_label}\ [\mathrm{{GeV}}]$")
ax.set_ylabel(get_variable_label("BR_mu_e"))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*X_LIMITS)
ax.set_ylim(*Y_LIMITS)


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
