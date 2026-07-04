import numpy as np
from numpy.linalg import LinAlgError, svd

from constants import MW_exp, mt_exp, vphi


# ---------------------------------------------------------
# Higgs diphoton signal strength
# ---------------------------------------------------------


def f_loop(rho):
    if rho >= 1.0:
        return np.arcsin(1.0 / np.sqrt(rho)) ** 2

    root = np.sqrt(1.0 - rho)
    log_term = np.log((1.0 + root) / (1.0 - root))
    return -0.25 * (log_term - 1j * np.pi) ** 2


def F_0(rho):
    return -rho * (1.0 - rho * f_loop(rho))


def F_1_half(rho):
    return 2.0 * rho * (1.0 + (1.0 - rho) * f_loop(rho))


def F_1(rho):
    return -(2.0 + 3.0 * rho + 3.0 * rho * (2.0 - rho) * f_loop(rho))


def diphoton_signal_strength(mh, mCH_array, g_hCHCH_array, mt, mW, v, ahtt=1.0, ahww=1.0):
    if mh <= 0.0 or not np.isfinite(mh):
        return np.nan

    nc = 3.0
    qt = 2.0 / 3.0

    rho_t = 4.0 * mt**2 / mh**2
    rho_w = 4.0 * mW**2 / mh**2

    amp_top_sm = nc * qt**2 * F_1_half(rho_t)
    amp_w_sm = F_1(rho_w)
    amp_sm = amp_top_sm + amp_w_sm

    amp_ch_bsm = 0j
    for m_ch, g_ch in zip(mCH_array, g_hCHCH_array):
        if m_ch > 0.0 and np.isfinite(m_ch) and np.isfinite(g_ch):
            rho_ch = 4.0 * m_ch**2 / mh**2
            amp_ch_bsm += (v / (2.0 * m_ch**2)) * g_ch * F_0(rho_ch)

    amp_bsm = ahtt * amp_top_sm + ahww * amp_w_sm + amp_ch_bsm
    return float(ahtt**2 * np.abs(amp_bsm) ** 2 / np.abs(amp_sm) ** 2)


# ---------------------------------------------------------
# Scalar observables
# ---------------------------------------------------------


def _invalid_scalar_result():
    return [
        np.full(3, np.nan),
        np.full(2, np.nan),
        np.full(3, np.nan),
        np.full(2, np.nan),
        np.identity(3),
        np.identity(3),
        np.nan,
    ], 1e20


def scalar_sector_observables(MI_even, MV_even, MI_odd, M_charged, gp1, gp2, alpha):
    try:
        u_even, s_even, _ = svd(MI_even)
        u_vis, s_vis, _ = svd(MV_even)
        u_odd, s_odd, _ = svd(MI_odd)
        u_charged, s_charged, _ = svd(M_charged)
    except (LinAlgError, ValueError, FloatingPointError):
        return _invalid_scalar_result()

    if not (
        np.all(np.isfinite(s_even))
        and np.all(np.isfinite(s_vis))
        and np.all(np.isfinite(s_odd))
        and np.all(np.isfinite(s_charged))
        and np.all(s_even > 0.0)
        and np.all(s_vis > 0.0)
        and np.all(s_odd > 0.0)
        and np.all(s_charged > 0.0)
    ):
        return _invalid_scalar_result()

    order_even = np.argsort(s_even)
    order_vis = np.argsort(s_vis)
    order_odd = np.argsort(s_odd)
    order_charged = np.argsort(s_charged)

    rr = u_even[:, order_even]
    ri = u_odd[:, order_odd]

    mR = np.sqrt(np.sort(s_even))
    mH = np.sqrt(np.sort(s_vis))
    mI = np.sqrt(np.sort(s_odd))
    mCH = np.sqrt(np.sort(s_charged))

    aww_model = np.cos(alpha)
    ahtt_model = np.cos(alpha)
    Rgamma = diphoton_signal_strength(
        mH[0],
        mCH,
        np.array([gp1, gp2]),
        mt_exp,
        MW_exp,
        vphi,
        ahtt_model,
        aww_model,
    )

    return [mR, mH, mI, mCH, rr, ri, Rgamma], 0.0
