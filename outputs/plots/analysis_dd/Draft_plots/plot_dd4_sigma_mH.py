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

from plot_template import c0, c1, c3, c4, c5, c6, c_reg1, c_esc1

DATA_PATH = ROOT / "outputs" / "analyses" / "analysis_dd" / "data" / "scans" / "scan_lfv.dat"
OUTPUT_PATH = Path(__file__).with_name("DD_mH.pdf")
STYLE_PATH = ROOT / "styles" / "paper_style_colorbar.mplstyle"

LZ = 7.413460268587916e-47
XENONNT = 4.0920440344790904e-47
DARKSIDE = 1.9212420771126029e-47
DARWIN = 5.895523679499314e-48
ARGO = 3.6753483466448516e-48
PURE_EW_SIGMA = 2.434770333230314e-47

YUKAWA_LIMIT = np.sqrt(4.0 * np.pi)
POINT_SIZE = 9
CROSS_SIZE = 13
X_LIMITS = (1.0e2, 1.0e8)
Y_LIMITS = (9.0e-49, 2.0e-45)


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
y_values = sigma_si
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
fig, ax = plt.subplots(figsize=(8.0, 5.0))


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
ax.axhline(XENONNT, color=c1, linestyle="-.", linewidth=1.3, zorder=3)
ax.axhline(DARKSIDE, color=c3, linestyle="-.", linewidth=1.3, zorder=3)
ax.axhline(DARWIN, color=c4, linestyle="-.", linewidth=1.3, zorder=3)
ax.axhline(ARGO, color=c0, linestyle="-.", linewidth=1.3, zorder=3)
ax.fill_between(
    X_LIMITS,
    LZ,
    1.0e-40,
    color=c_reg1,
    alpha=0.28,
    zorder=0,
    rasterized=True,
)


# === TEXTS, ANNOTATIONS AND MARKERS ===
line_label_style = {
    "va": "bottom",
    "ha": "right",
    "fontsize": 11,
    "zorder": 10,
}
label_x = 8.0e7
ax.annotate(
    r"LUX-ZEPLIN $(4.5\mathrm{t}\times \mathrm{y})$",
    xy=(label_x, LZ),
    xytext=(0, 3),
    textcoords="offset points",
    color=c_esc1,
    **line_label_style,
)
ax.annotate(
    r"XENONnT $(20\mathrm{t}\times \mathrm{y})$",
    xy=(label_x, XENONNT),
    xytext=(0, 3),
    textcoords="offset points",
    color=c1,
    **line_label_style,
)
ax.annotate(
    r"DARWIN $(200\mathrm{t}\times \mathrm{y})$",
    xy=(label_x, DARWIN),
    xytext=(0, 3),
    textcoords="offset points",
    color=c4,
    **line_label_style,
)
ax.annotate(
    r"Minimal fermionic triplet",
    xy=(label_x, PURE_EW_SIGMA),
    xytext=(0, 3),
    textcoords="offset points",
    color="red",
    **line_label_style,
)
ax.annotate(
    r"DarkSide-20k $(200\mathrm{t}\times \mathrm{y})$",
    xy=(label_x, DARKSIDE),
    xytext=(0, -3),
    textcoords="offset points",
    color=c3,
    va="top",
    ha="right",
    fontsize=11,
    zorder=10,
)
ax.annotate(
    r"ARGO $(3000\mathrm{t}\times \mathrm{y})$",
    xy=(label_x, ARGO),
    xytext=(0, 3),
    textcoords="offset points",
    color=c0,
    **line_label_style,
)


# === AXES AND LEGEND ===
ax.set_xlabel(r"$m_H\ [{\rm GeV}]$")
ax.set_ylabel(r"$\sigma_{\rm SI}\ [{\rm cm}^2]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*X_LIMITS)
ax.set_ylim(*Y_LIMITS)


# === EXPORT ===
fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
