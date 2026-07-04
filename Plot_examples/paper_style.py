import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

### COLORES PERSONALIZADOS

c_reg1 = (210/255, 57/255, 49/255)
c_reg2 = (65/255, 65/255, 239/255)
c_reg3 = (89/255, 169/255, 69/255)

c_esc1 = (150/255, 40/255, 29/255)
c_esc2 = (47/255, 45/255, 241/255)
c_esc3 = (80/255, 153/255, 49/255)

c0 = (18/255, 78/255, 203/255)
c1 = (229/255, 63/255, 158/255)
c2 = (64/255, 188/255, 42/255)
c3 = (52/255, 160/255, 243/255)
c4 = (165/255, 42/255, 214/255)
c5 = (220/255, 167/255, 35/255)
c6 = (41/255, 214/255, 164/255)

CB_STYLES = {
    'lat': {'kwargs': {'location': 'right', 'fraction': 0.09, 'pad': 0.01, 'extend': 'both'}},
    'top':     {'kwargs': {'location': 'top', 'fraction': 0.06, 'pad': 0.01, 'extend': 'both'}},
    'in':{'kwargs': {'orientation': 'horizontal', 'extend': 'both'}}
}

def sp_scatter(ax, x, y, allow_cond=None, c_theme=c0, x_lab=r'$\alpha$ [$\mu$s]', y_lab=r'$\ell(\alpha)$ [pb]', lw=0.4):

    # Lógica de las máscaras
    if allow_cond is None:
        # Si no hay condición, todos los puntos son válidos (Arreglo de Trues)
        allow_mask = np.ones(len(x), dtype=bool) 
    else:
        # Si pasaste una condición, la usamos tal cual
        allow_mask = allow_cond

    # Ahora la negación siempre funcionará sobre un arreglo de NumPy
    excluded_mask = ~allow_mask

    # Puntos permitidos (a color)
    sc_main = ax.scatter(
        x[allow_mask], y[allow_mask],
        color=c_theme,
        edgecolor='black', linewidth=lw,
        alpha=0.9, s=15, zorder=2, rasterized=True
    )

    # Puntos excluidos (fondo gris)
    ax.scatter(
        x[excluded_mask], y[excluded_mask],
        color='gray', 
        edgecolor='black', linewidth=lw,
        alpha=0.5, s=7, zorder=1, rasterized=True
    )

    offset_text_y = ax.yaxis.get_offset_text()
    offset_text_x = ax.xaxis.get_offset_text()
    # Coordenadas (x, y). Aumenta el 1.05 para separarlo más hacia arriba.
    offset_text_y.set_position((-0.1, 1.5))
    offset_text_x.set_position((1.05, 0))

    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    
    return sc_main

def cb_scatter(ax, x, y, c_val, allow_cond=None, c_theme='viridis', cb_pos='in', cb_loc='upper right', x_lab=r'$\alpha$ [$\mu$s]', y_lab=r'$\ell(\alpha)$ [pb]', c_lab=r'$c$ $\left[\mathrm{pb}/\mu\mathrm{s}\right]$', lw=0.4, lab_pos=-0.1):

    # Configuración del mapa de color
    cmap = plt.get_cmap(c_theme).copy()
    cmap.set_under('0.65')
    norm = colors.Normalize(vmin=c_val.min(), vmax=c_val.max())

    # Lógica de las máscaras
    if allow_cond is None:
        # Si no hay condición, todos los puntos son válidos (Arreglo de Trues)
        allow_mask = np.ones(len(x), dtype=bool) 
    else:
        # Si pasaste una condición, la usamos tal cual
        allow_mask = allow_cond

    # Ahora la negación siempre funcionará sobre un arreglo de NumPy
    excluded_mask = ~allow_mask
    norme = colors.Normalize(vmin=c_val[allow_mask].min(), vmax=c_val[allow_mask].max())
    # Puntos permitidos (a color)
    sc_main = ax.scatter(
        x[allow_mask], y[allow_mask],
        c=c_val[allow_mask], cmap=cmap, norm=norme,
        edgecolor='black', linewidth=lw,
        alpha=0.9, s=15, zorder=2, rasterized=True
    )

    # Puntos excluidos (fondo gris)
    ax.scatter(
        x[excluded_mask], y[excluded_mask],
        color='gray', 
        edgecolor='black', linewidth=lw,
        alpha=0.5, s=7, zorder=1, rasterized=True
    )

    fig = ax.figure # Extraemos la figura a partir del eje
    style = CB_STYLES[cb_pos]['kwargs']
    
    if cb_pos == 'in':
        cax = inset_axes(ax, width="35%", height="5%", loc=cb_loc, borderpad=2.5)
        #cax = inset_axes(ax, width="20%", height="5%", loc=cb_loc, borderpad=1.5)
        cb = fig.colorbar(sc_main, cax=cax, **style)
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.tick_params(labelsize=16)
        cb.ax.set_title(c_lab, fontsize=16, pad=7)
    else:
        cb = fig.colorbar(sc_main, ax=ax, **style)
        if cb_pos == 'lat':
            cb.set_label(c_lab, fontsize=22)
        elif cb_pos == 'top':
            cb.ax.set_ylabel(c_lab, fontsize=18, rotation=0, va='center', ha='left')
            cb.ax.yaxis.set_label_coords(lab_pos, 1.8) 

    # Formato científico para todas
    cb.update_normal(sc_main)
    cb.formatter.set_scientific(True)

    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    
    return sc_main

def del_reg(ax, l_val , r_val, v_min=0, v_max=15, l_col=c_esc1, r_col=c_reg1, ori='h', l_lab=r'Exp. limit', z_ord=4):

    if ori=='h':
        lin = ax.hlines(l_val, xmin=v_min, xmax=v_max, color=l_col, linestyle='-', linewidth=0.9, zorder=z_ord, label=l_lab)
    
        ax.fill_between(
        x=np.linspace(v_min, v_max, 10),
        y1=l_val,
        y2=r_val,
        color=r_col, 
        alpha=0.3,    
        zorder=z_ord, rasterized=True    
        )
        
    if ori=='v':
        lin = ax.vlines(x=l_val, ymin=v_min, ymax=v_max, color=l_col, linestyle='-', linewidth=0.8, zorder=z_ord, label=l_lab)
        
        ax.axvspan(
        xmin=l_val, 
        xmax=r_val, 
        color=r_col,
        alpha=0.3, 
        zorder=z_ord, rasterized=True
        )
    
    return lin