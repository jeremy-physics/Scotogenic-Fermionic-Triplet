from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D


# === GLOBAL PLOT CONFIGURATION ===
ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = ROOT / "src"
PLOT_TEMPLATE_PATH = ROOT / "scripts"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PLOT_TEMPLATE_PATH) not in sys.path:
    sys.path.insert(0, str(PLOT_TEMPLATE_PATH))

from plot_template import c0, c1, c2, c3, c4, c5, c6, c_reg1, c_reg2, c_reg3, c_esc1, c_esc2, c_esc3
from constants import (
    BR_mu_3e_proy,
    BR_mu_egamma_lim,
    BR_mu_egamma_proy,
    CR_muAl_e_proy,
)
from variables import get_variable_label

DATA_PATH = ROOT / "outputs" / "analyses" / "analysis_dd" / "data" / "scans" / "scan_lfv.dat"
OUTPUT_PATH = Path(__file__).with_name("fig10_lfv_mu_e_cr.pdf")
STYLE_PATH = ROOT / "styles" / "paper_style_colorbar.mplstyle"

LZ = 8.78658722623048e-47
YUKAWA_LIMIT = np.sqrt(4.0 * np.pi)
POINT_SIZE = 12
POINT_EDGE_COLOR = "none"
POINT_LINEWIDTH = 0.0
COLOR_LIMITS = (-20.0, -12.0)
X_LIMITS = (3e-17, 1e-12)
Y_LIMITS = (1e-20, 5e-16)
LOG10_BR_MU_3E_PROY = np.log10(BR_mu_3e_proy)


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
BR_mu_3e = scan["BR_mu_3e"]
CR_muAL_e = scan["CR_muAL_e"]
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

with np.errstate(divide="ignore", invalid="ignore"):
    log10_BR_mu_3e = np.log10(BR_mu_3e)

mask_yukawa = np.isfinite(yXi_max) & (yXi_max <= YUKAWA_LIMIT)
mask_base = (
    mask_yukawa
    & np.isfinite(BR_mu_e)
    & np.isfinite(CR_muAL_e)
    & np.isfinite(log10_BR_mu_3e)
    & np.isfinite(sigma_si)
    & (BR_mu_e > 0.0)
    & (CR_muAL_e > 0.0)
    & (BR_mu_3e > 0.0)
    & (sigma_si > 0.0)
)
mask_lz = mask_base & (sigma_si > LZ)
mask_color = mask_base & ~mask_lz


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
cmap = plt.get_cmap("summer").copy()
cmap.set_under('lightgrey')
cmap.set_over('lightgrey')
norm = colors.Normalize(vmin=COLOR_LIMITS[0], vmax=COLOR_LIMITS[1])
ax.scatter(
    BR_mu_e[mask_lz],
    CR_muAL_e[mask_lz],
    marker="o",
    color='grey',
    edgecolors=POINT_EDGE_COLOR,
    linewidths=POINT_LINEWIDTH,
    s=POINT_SIZE,
    alpha=0.5,
    rasterized=True,
    zorder=1,
    #label=r"$\sigma_{\rm SI}>{\rm LZ}$",
)
scatter = ax.scatter(
    BR_mu_e[mask_color],
    CR_muAL_e[mask_color],
    c=log10_BR_mu_3e[mask_color],
    cmap=cmap,
    norm=norm,
    marker="o",
    edgecolors=POINT_EDGE_COLOR,
    linewidths=POINT_LINEWIDTH,
    s=POINT_SIZE,
    alpha=1.0,
    rasterized=True,
    zorder=2,
)


# === LINES AND REGIONS ===
ax.axvspan(BR_mu_egamma_lim, X_LIMITS[1], color=c_reg1, alpha=0.18, zorder=0)
ax.axvline(BR_mu_egamma_lim, color=c_esc1, linestyle="-", linewidth=1.2)
ax.axvline(BR_mu_egamma_proy, color=c_esc1, linestyle='-.', linewidth=1.5, label=r"MEG-II prospect")
ax.axhline(CR_muAl_e_proy, color=c0, linestyle="-.", linewidth=1.5, label=r"COMET prospect")


# === TEXTS, ANNOTATIONS AND MARKERS ===
ax.text(
    #np.sqrt(BR_mu_egamma_lim * X_LIMITS[1]),
    #np.sqrt(Y_LIMITS[0] * Y_LIMITS[1]),
    2.5e-13,
    2.5e-19,
    r"excluded by MEG-II",
    color=c_esc1,
    rotation=90,
    ha="center",
    va="center",
    fontsize=16,
    bbox={
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.9,
    },
    zorder=10,
)


# === AXES AND LEGEND ===
ax.set_xlabel(get_variable_label("BR_mu_e"))
ax.set_ylabel(get_variable_label("CR_muAL_e"))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*X_LIMITS)
ax.set_ylim(*Y_LIMITS)
handles, _ = ax.get_legend_handles_labels()
handles.append(
    Line2D(
        [0],
        [0],
        color=c1,
        linestyle="-",
        linewidth=1.7,
        label=r"Mu3e prospect",
    )
)
legend = ax.legend(handles=handles, fontsize=14, loc="upper left", frameon=True)
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_boxstyle("Round,pad=0.1")

cbar_ax = ax.inset_axes([0.0, 1.03, 1.0, 0.045], transform=ax.transAxes)
cbar = fig.colorbar(scatter, cax=cbar_ax, orientation="horizontal", extend="both")
cbar.set_label(r"$\log_{10}{\rm BR}(\mu\to 3e)$", fontsize=16)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.tick_params(direction="in", labelsize=12, pad=2)
if COLOR_LIMITS[0] <= LOG10_BR_MU_3E_PROY <= COLOR_LIMITS[1]:
    cbar.ax.axvline(LOG10_BR_MU_3E_PROY, color=c1, linewidth=1.7, zorder=10)
else:
    x_marker = 1.0 if LOG10_BR_MU_3E_PROY > COLOR_LIMITS[1] else 0.0
    cbar.ax.plot(
        [x_marker, x_marker],
        [0.0, 1.0],
        color=c1,
        linewidth=1.7,
        transform=cbar.ax.transAxes,
        clip_on=False,
        zorder=10,
    )


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
