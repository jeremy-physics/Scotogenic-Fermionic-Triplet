from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]
STYLE_DIR = ROOT / "styles"


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


CB_STYLES = {
    "lat": {
        "kwargs": {
            "location": "right",
            "fraction": 0.09,
            "pad": 0.01,
            "extend": "both",
        }
    },
    "top": {
        "kwargs": {
            "location": "top",
            "fraction": 0.06,
            "pad": 0.01,
            "extend": "both",
        }
    },
    "in": {
        "kwargs": {
            "orientation": "horizontal",
            "extend": "both",
        }
    },
}


def apply_paper_style(style="colorbar"):
    style_files = {
        "colorbar": STYLE_DIR / "paper_style_colorbar.mplstyle",
        "paper": STYLE_DIR / "paper_style.mplstyle",
    }
    if style not in style_files:
        print(f"[WARN] Unknown paper style '{style}'. Using matplotlib defaults.")
        style_path = None
    else:
        style_path = style_files[style]
        if style_path.is_file():
            plt.style.use(str(style_path))
        else:
            print(f"[WARN] Matplotlib style file not found: {style_path}")
            style_path = None

    plt.rcParams["text.usetex"] = True
    plt.rc("font", family="serif")
    return style_path


def _scatter_arrays(x, y, allow_cond):
    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape.")

    if allow_cond is None:
        allow_mask = np.ones(x_values.shape, dtype=bool)
    else:
        allow_mask = np.asarray(allow_cond, dtype=bool)
        if allow_mask.shape != x_values.shape:
            raise ValueError("allow_cond must have the same shape as x and y.")
    return x_values, y_values, allow_mask


def sp_scatter(
    ax,
    x,
    y,
    allow_cond=None,
    c_theme=c0,
    x_lab=r"$x$",
    y_lab=r"$y$",
    excluded_scatter_kwargs=None,
    **scatter_kwargs,
):
    x_values, y_values, allow_mask = _scatter_arrays(x, y, allow_cond)
    excluded_mask = ~allow_mask

    main_kwargs = {
        "color": c_theme,
        "edgecolors": "none",
        "linewidths": 0.0,
        "alpha": 0.9,
        "s": 20,
        "zorder": 2,
    }
    main_kwargs.update(scatter_kwargs)
    sc_main = ax.scatter(x_values[allow_mask], y_values[allow_mask], **main_kwargs)

    if np.any(excluded_mask):
        excluded_kwargs = {
            "color": "gray",
            "edgecolors": "none",
            "linewidths": 0.0,
            "alpha": 0.5,
            "s": main_kwargs.get("s", 20),
            "zorder": 1,
        }
        excluded_kwargs.update(excluded_scatter_kwargs or {})
        ax.scatter(
            x_values[excluded_mask],
            y_values[excluded_mask],
            **excluded_kwargs,
        )

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    return sc_main


def cb_scatter(
    ax,
    x,
    y,
    c_val,
    allow_cond=None,
    c_theme="viridis",
    cb_pos="in",
    cb_loc="upper right",
    x_lab=r"$x$",
    y_lab=r"$y$",
    c_lab=r"$c$",
    norm=None,
    add_colorbar=True,
    excluded_scatter_kwargs=None,
    **scatter_kwargs,
):
    x_values, y_values, allow_mask = _scatter_arrays(x, y, allow_cond)
    color_values = np.asarray(c_val)
    if color_values.shape != x_values.shape:
        raise ValueError("c_val must have the same shape as x and y.")
    if cb_pos not in CB_STYLES:
        raise ValueError(f"Unknown colorbar position: {cb_pos}")

    finite_colors = color_values[np.isfinite(color_values)]
    if finite_colors.size == 0:
        raise ValueError("c_val does not contain finite values.")

    cmap = plt.get_cmap(c_theme).copy()
    cmap.set_under("0.65")
    if norm is None:
        norm = colors.Normalize(
            vmin=np.min(finite_colors),
            vmax=np.max(finite_colors),
        )

    excluded_mask = ~allow_mask
    main_kwargs = {
        "c": color_values[allow_mask],
        "cmap": cmap,
        "norm": norm,
        "edgecolors": "none",
        "linewidths": 0.0,
        "alpha": 0.9,
        "s": 20,
        "zorder": 2,
    }
    main_kwargs.update(scatter_kwargs)
    sc_main = ax.scatter(x_values[allow_mask], y_values[allow_mask], **main_kwargs)

    if np.any(excluded_mask):
        excluded_kwargs = {
            "color": "gray",
            "edgecolors": "none",
            "linewidths": 0.0,
            "alpha": 0.5,
            "s": main_kwargs.get("s", 20),
            "zorder": 1,
        }
        excluded_kwargs.update(excluded_scatter_kwargs or {})
        ax.scatter(
            x_values[excluded_mask],
            y_values[excluded_mask],
            **excluded_kwargs,
        )

    if add_colorbar:
        fig = ax.figure
        style_kwargs = dict(CB_STYLES[cb_pos]["kwargs"])
        if cb_pos == "in":
            cax = inset_axes(
                ax,
                width="35%",
                height="5%",
                loc=cb_loc,
                borderpad=2.5,
            )
            colorbar = fig.colorbar(sc_main, cax=cax, **style_kwargs)
            colorbar.ax.xaxis.set_ticks_position("bottom")
            colorbar.ax.tick_params(labelsize=14)
            colorbar.ax.set_title(c_lab, fontsize=14, pad=7)
        else:
            colorbar = fig.colorbar(sc_main, ax=ax, **style_kwargs)
            if cb_pos == "lat":
                colorbar.set_label(c_lab, fontsize=22)
            else:
                colorbar.ax.set_ylabel(
                    c_lab,
                    fontsize=18,
                    rotation=0,
                    va="center",
                    ha="left",
                )
                colorbar.ax.yaxis.set_label_coords(-0.1, 1.8)

        colorbar.update_normal(sc_main)
        if hasattr(colorbar.formatter, "set_scientific"):
            colorbar.formatter.set_scientific(True)

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    return sc_main


def del_reg(
    ax,
    l_val,
    r_val,
    v_min=0,
    v_max=15,
    l_col=c_esc1,
    r_col=c_reg1,
    ori="h",
    l_lab=r"Exp. limit",
    z_ord=4,
    line_kwargs=None,
    fill_kwargs=None,
):
    line_style = {
        "color": l_col,
        "linestyle": "-",
        "linewidth": 0.9,
        "zorder": z_ord,
        "label": l_lab,
    }
    line_style.update(line_kwargs or {})
    region_style = {"color": r_col, "alpha": 0.3, "zorder": z_ord}
    region_style.update(fill_kwargs or {})

    if ori == "h":
        line = ax.hlines(l_val, xmin=v_min, xmax=v_max, **line_style)
        boundary = np.asarray(l_val)
        sample_count = boundary.size if boundary.ndim > 0 else 100
        x_values = np.linspace(v_min, v_max, sample_count)
        ax.fill_between(x_values, l_val, r_val, **region_style)
        return line

    if ori == "v":
        if not line_kwargs or "linewidth" not in line_kwargs:
            line_style["linewidth"] = 0.8
        line = ax.vlines(l_val, ymin=v_min, ymax=v_max, **line_style)
        ax.axvspan(l_val, r_val, **region_style)
        return line

    raise ValueError("ori must be 'h' or 'v'.")


def style_legend(leg):
    if leg is None:
        return None
    leg.get_frame().set_linewidth(1)
    leg.get_frame().set_alpha(1)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_boxstyle("Round,pad=0.1")
    return leg
