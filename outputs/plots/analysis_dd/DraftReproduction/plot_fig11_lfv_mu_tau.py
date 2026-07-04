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

from plot_template import (
    c0, c1, c2, c3, c4, c5, c6, c_reg1, c_reg2, c_reg3, c_esc1, c_esc2, c_esc3
)
from constants import (
    BR_mu_egamma_lim,
    BR_mu_egamma_proy,
    BR_tau_mugamma_lim,
    BR_tau_mugamma_proy,
)
from variables import get_variable_label

DATA_PATH = ROOT / "outputs" / "analyses" / "analysis_dd" / "data" / "scans" / "scan_lfv.dat"
OUTPUT_PATH = Path(__file__).with_name("fig11_lfv_mu_tau.pdf")
STYLE_PATH = ROOT / "styles" / "paper_style_colorbar.mplstyle"

LZ = 8.78658722623048e-47
YUKAWA_LIMIT = np.sqrt(4.0 * np.pi)
POINT_SIZE = 9
CROSS_SIZE = 13
POINT_EDGE_COLOR = "none"
POINT_LINEWIDTH = 0.0
X_LIMITS = (1e-20, 1e-12)
Y_LIMITS = (1e-20, 3e-8)


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
BR_mu_e = scan["BR_mu_e"]
BR_tau_mu = scan["BR_tau_mu"]
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
    & np.isfinite(BR_mu_e)
    & np.isfinite(BR_tau_mu)
    & np.isfinite(sigma_si)
    & (BR_mu_e > 0.0)
    & (BR_tau_mu > 0.0)
    & (sigma_si > 0.0)
)
mask_lz = mask_base & (sigma_si > LZ)
mask_kept = mask_base & ~mask_lz
mask_allowed = mask_kept #& (BR_mu_e <= BR_mu_egamma_lim) & (BR_tau_mu <= BR_tau_mugamma_lim)
mask_excluded = mask_kept & ~mask_allowed


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
    BR_mu_e[mask_lz],
    BR_tau_mu[mask_lz],
    marker="o",
    color='lightgrey',
    edgecolors=POINT_EDGE_COLOR,
    linewidths=POINT_LINEWIDTH,
    s=POINT_SIZE,
    alpha=0.6,
    rasterized=True,
    zorder=0,
    #label=r"$\sigma_{\rm SI}>{\rm LZ}$",
)
ax.scatter(
    BR_mu_e[mask_excluded],
    BR_tau_mu[mask_excluded],
    marker="x",
    color=c5,
    linewidths=0.6,
    s=CROSS_SIZE,
    alpha=0.75,
    rasterized=True,
    zorder=1,
    #label=r"Current limit excluded",
)
ax.scatter(
    BR_mu_e[mask_allowed],
    BR_tau_mu[mask_allowed],
    marker="o",
    color=c2,
    edgecolors=POINT_EDGE_COLOR,
    linewidths=POINT_LINEWIDTH,
    s=POINT_SIZE,
    alpha=0.9,
    rasterized=True,
    zorder=2,
    #label=r"Current limit allowed",
)


# === LINES AND REGIONS ===
ax.axvspan(BR_mu_egamma_lim, X_LIMITS[1], color=c_reg1, alpha=0.16, zorder=0)
#ax.axhspan(BR_tau_mugamma_lim, Y_LIMITS[1], color=c5, alpha=0.16, zorder=0)
ax.axvline(BR_mu_egamma_lim, color=c_esc1, linestyle="-", linewidth=1.2)
#ax.axhline(BR_tau_mugamma_lim, color=c0, linestyle="-", linewidth=1.2, label=r"$\tau\to\mu\gamma$ limit")
ax.axvline(BR_mu_egamma_proy, color=c1, linestyle="-.", linewidth=1.5, label=r"MEG-II prospect")
ax.axhline(BR_tau_mugamma_proy, color=c0, linestyle="-.", linewidth=1.5, label=r"superKEKB/Belle-II prospect")


# === TEXTS, ANNOTATIONS AND MARKERS ===

ax.text(
    #np.sqrt(BR_mu_egamma_lim * X_LIMITS[1]),
    #np.sqrt(Y_LIMITS[0] * Y_LIMITS[1]),
    3e-13,
    2.5e-17,
    r"excluded by MEG-II",
    color=c_esc1,
    rotation=90,
    ha="center",
    va="center",
    fontsize=13,
    bbox={
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.5,
    },
    zorder=10,
)

# === AXES AND LEGEND ===
ax.set_xlabel(get_variable_label("BR_mu_e"))
ax.set_ylabel(get_variable_label("BR_tau_mu"))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*X_LIMITS)
ax.set_ylim(*Y_LIMITS)
legend = ax.legend(fontsize=12, loc="upper right", frameon=True)
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_boxstyle("Round,pad=0.1")


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
