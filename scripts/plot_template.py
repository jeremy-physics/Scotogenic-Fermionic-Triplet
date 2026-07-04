from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import transforms


# === USER/AGENT CONFIGURATION START ===

# === GLOBAL PLOT CONFIGURATION ===

ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_NAME = "MyAnalysis"
PLOTSET_NAME = "MyPlotSet"

INPUT_FILE = (
    ROOT
    / "outputs"
    / "plots"
    / ANALYSIS_NAME
    / PLOTSET_NAME
    / "input_data.dat"
)

OUTPUT_DIR = ROOT / "outputs" / "plots" / ANALYSIS_NAME / PLOTSET_NAME
OUTPUT_FILE = OUTPUT_DIR / "plot_template.pdf"

STYLE_FILE = ROOT / "styles" / "paper_style_colorbar.mplstyle"

USE_TEX = True
SHOW_FIGURE = False
SAVE_FIGURE = True
OVERWRITE_FIGURE = False

FIGSIZE = (9.0, 5.0)
AX_RECT = [0.11, 0.13, 0.70, 0.80]

X_KEY = "x"
Y_KEY = "y"
C_KEY = "cost_function"

X_LABEL = r"$x$"
Y_LABEL = r"$y$"
C_LABEL = r"$\mathcal{C}$"

X_SCALE = None
Y_SCALE = None

X_LIMITS = None
Y_LIMITS = None

POINT_SIZE = 16
POINT_ALPHA = 0.85
POINT_LINEWIDTH = 0.0
POINT_EDGE_COLOR = "none"

COLORMAP = "viridis"


# === COLOR PALETTE ===

c_reg1 = (210 / 255, 57 / 255, 49 / 255)
c_reg2 = (65 / 255, 65 / 255, 239 / 255)
c_reg3 = (89 / 255, 169 / 255, 69 / 255)

c_esc1 = (150 / 255, 40 / 255, 29 / 255)
c_esc2 = (47 / 255, 45 / 255, 241 / 255)
c_esc3 = (80 / 255, 153 / 255, 49 / 255)

c0 = (18 / 255, 78 / 255, 203 / 255)
c1 = (229 / 255, 63 / 255, 158 / 255)
c2 = (64 / 255, 188 / 255, 42 / 255)
c3 = (52 / 255, 160 / 255, 243 / 255)
c4 = (165 / 255, 42 / 255, 214 / 255)
c5 = (220 / 255, 167 / 255, 35 / 255)
c6 = (41 / 255, 214 / 255, 164 / 255)

COLOR_PALETTE = (c0, c1, c2, c3, c4, c5, c6)

# === USER/AGENT CONFIGURATION END ===


def require_columns(data, columns):
    missing = [col for col in columns if col not in data.dtype.names]
    if missing:
        raise KeyError(
            f"Missing required columns in {INPUT_FILE}: {missing}. "
            f"Available columns: {data.dtype.names}"
        )


def main():
    # === DATA LOADING AND PREPARATION ===

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    data = np.genfromtxt(INPUT_FILE, names=True)

    if data.dtype.names is None:
        raise ValueError(
            f"Input file has no named columns. Expected a header line in: {INPUT_FILE}"
        )

    require_columns(data, [X_KEY, Y_KEY, C_KEY])

    x = np.atleast_1d(data[X_KEY])
    y = np.atleast_1d(data[Y_KEY])
    c_val = np.atleast_1d(data[C_KEY])

    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c_val)

    if not np.any(valid_mask):
        raise ValueError("No valid finite points available for plotting.")

    x_plot = x[valid_mask]
    y_plot = y[valid_mask]
    c_plot = c_val[valid_mask]


    # === STYLE ===

    if STYLE_FILE.exists():
        plt.style.use(STYLE_FILE)

    if USE_TEX:
        matplotlib.rcParams["text.usetex"] = True

    plt.rc("font", family="serif")


    # === FIGURE AND AXES ===

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AX_RECT)

    trans_xaxes_ydata = transforms.blended_transform_factory(
        ax.transAxes,
        ax.transData,
    )


    # === PLOTS ===

    cmap = plt.get_cmap(COLORMAP).copy()

    norm = colors.Normalize(
        vmin=np.nanmin(c_plot),
        vmax=np.nanmax(c_plot),
    )

    sc = ax.scatter(
        x_plot,
        y_plot,
        c=c_plot,
        cmap=cmap,
        norm=norm,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=POINT_LINEWIDTH,
        rasterized=True,
        zorder=2,
    )

    cb = fig.colorbar(
        sc,
        ax=ax,
        location="right",
        fraction=0.08,
        pad=0.02,
    )

    cb.set_label(C_LABEL)
    cb.update_normal(sc)


    # === LINES AND REGIONS ===

    # Example line:
    # ax.axhline(
    #     y=1.0,
    #     color=c_esc1,
    #     linestyle="-",
    #     linewidth=1.0,
    #     zorder=4,
    #     label=r"Reference",
    # )

    # Example shaded region:
    # x_region = np.linspace(np.nanmin(x_plot), np.nanmax(x_plot), 200)
    # ax.fill_between(
    #     x_region,
    #     1.0,
    #     np.nanmax(y_plot),
    #     color=c_reg1,
    #     alpha=0.25,
    #     linewidth=0.0,
    #     rasterized=True,
    #     zorder=1,
    # )


    # === TEXTS, ANNOTATIONS AND MARKERS ===

    # Example annotation:
    # ax.text(
    #     0.05,
    #     0.95,
    #     r"ELBAPH2.0",
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=14,
    #     zorder=10,
    # )


    # === AXES AND LEGEND ===

    if X_SCALE is not None:
        ax.set_xscale(X_SCALE)

    if Y_SCALE is not None:
        ax.set_yscale(Y_SCALE)

    if X_LIMITS is not None:
        ax.set_xlim(*X_LIMITS)

    if Y_LIMITS is not None:
        ax.set_ylim(*Y_LIMITS)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(
            loc="best",
            frameon=True,
            fontsize=11,
        )
        leg.get_frame().set_linewidth(1.0)
        leg.get_frame().set_alpha(1.0)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_boxstyle("Round,pad=0.1")


    # === EXPORT ===

    if SAVE_FIGURE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if OUTPUT_FILE.exists() and not OVERWRITE_FIGURE:
            raise FileExistsError(
                f"Output file already exists: {OUTPUT_FILE}. "
                "Change OUTPUT_FILE or set OVERWRITE_FIGURE=True."
            )

        fig.savefig(
            OUTPUT_FILE,
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
