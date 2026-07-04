import numpy as np

from constants import (
    BR_mu_3e_proy,
    BR_mu_egamma_lim,
    BR_mu_egamma_proy,
    BR_tau_mugamma_lim,
    BR_tau_mugamma_proy,
    CR_muAl_e_proy,
    nu_sup_exp,
    nu_sup_inf,
)


_WARNED_MISSING_VARIABLES = set()


def hybrid_pen(x, x0, p=4.0):
    """Soft guide penalty for phenomenologically guided scans."""
    abs_x = np.abs(x) + 1e-20
    abs_x0 = np.abs(x0) + 1e-20

    ratio = abs_x / abs_x0
    cut = 0.1

    if ratio < cut:
        log_term = np.log10(ratio) / np.log10(cut)
        return log_term**p

    norm_factor = abs_x0 * (cut - 1.0)
    return ((abs_x - abs_x0) / norm_factor) ** 2.0


def _get_restriction_value(values, key, default=0.0):
    if key not in values and key not in _WARNED_MISSING_VARIABLES:
        print(
            f"[WARN] Restriction variable not found in values: {key}. "
            f"Using default {default}."
        )
        _WARNED_MISSING_VARIABLES.add(key)
    return values.get(key, default)


def compute_restriction_penalty(values, settings=None):
    sum_nu = _get_restriction_value(values, "sum_nu")
    neutrino_sum_lower_limit = nu_sup_inf * 1e-9
    neutrino_sum_upper_limit = nu_sup_exp * 1e-9

    pen_neutrino_sum_lower = 0.0
    pen_neutrino_sum_upper = 0.0
    if not np.isfinite(sum_nu):
        pen_neutrino_sum_upper = 1e20
    elif sum_nu < neutrino_sum_lower_limit:
        violation = neutrino_sum_lower_limit - sum_nu
        pen_neutrino_sum_lower = 1e12 * violation**2
    elif sum_nu > neutrino_sum_upper_limit:
        violation = sum_nu - neutrino_sum_upper_limit
        pen_neutrino_sum_upper = 1e12 * violation**2

    penalties = {
        "neutrino_mass_sum_lower": pen_neutrino_sum_lower,
        "neutrino_mass_sum_upper": pen_neutrino_sum_upper,
    }

    active_observables = getattr(settings, "ACTIVE_OBSERVABLES", []) if settings is not None else []
    lfv_penalties_are_active = (
        "lfv" in active_observables
        or "lepton_flavor_violation" in active_observables
    )

    if lfv_penalties_are_active:
        BR_mu_e = _get_restriction_value(values, "BR_mu_e", default=np.nan)
        BR_tau_mu = _get_restriction_value(values, "BR_tau_mu", default=np.nan)
        BR_mu_3e = _get_restriction_value(values, "BR_mu_3e", default=np.nan)
        CR_muAL_e = _get_restriction_value(values, "CR_muAL_e", default=np.nan)

        pen_BR_mu_e_limit = 0.0
        if not np.isfinite(BR_mu_e) or BR_mu_e < 0.0:
            pen_BR_mu_e_limit = 1e20
        elif BR_mu_e > BR_mu_egamma_lim:
            violation = BR_mu_e / BR_mu_egamma_lim - 1.0
            pen_BR_mu_e_limit = 1e20 * violation**2

        pen_BR_tau_mu_limit = 0.0
        if not np.isfinite(BR_tau_mu) or BR_tau_mu < 0.0:
            pen_BR_tau_mu_limit = 1e20
        elif BR_tau_mu > BR_tau_mugamma_lim:
            violation = BR_tau_mu / BR_tau_mugamma_lim - 1.0
            pen_BR_tau_mu_limit = 1e20 * violation**2

        pen_BR_mu_e_projection = 0.0
        if not np.isfinite(BR_mu_e) or BR_mu_e < 0.0:
            pen_BR_mu_e_projection = 1e20
        elif BR_mu_e < BR_mu_egamma_proy:
            pen_BR_mu_e_projection = hybrid_pen(BR_mu_e, BR_mu_egamma_proy)

        pen_BR_tau_mu_projection = 0.0
        if not np.isfinite(BR_tau_mu) or BR_tau_mu < 0.0:
            pen_BR_tau_mu_projection = 1e20
        elif BR_tau_mu < BR_tau_mugamma_proy:
            pen_BR_tau_mu_projection = hybrid_pen(BR_tau_mu, BR_tau_mugamma_proy)

        pen_BR_mu_3e_projection = 0.0
        if not np.isfinite(BR_mu_3e) or BR_mu_3e < 0.0:
            pen_BR_mu_3e_projection = 1e20
        elif BR_mu_3e < BR_mu_3e_proy:
            pen_BR_mu_3e_projection = hybrid_pen(BR_mu_3e, BR_mu_3e_proy)

        pen_CR_muAL_e_projection = 0.0
        if not np.isfinite(CR_muAL_e) or CR_muAL_e < 0.0:
            pen_CR_muAL_e_projection = 1e20
        elif CR_muAL_e < CR_muAl_e_proy:
            pen_CR_muAL_e_projection = hybrid_pen(CR_muAL_e, CR_muAl_e_proy)

        penalties.update(
            {
                #"BR_mu_e_limit": pen_BR_mu_e_limit,
                #"BR_tau_mu_limit": pen_BR_tau_mu_limit,
                #"BR_mu_e_projection": pen_BR_mu_e_projection,
                #"BR_tau_mu_projection": pen_BR_tau_mu_projection,
                #"BR_mu_3e_projection": pen_BR_mu_3e_projection,
                #"CR_muAL_e_projection": pen_CR_muAL_e_projection,
            }
        )

    return sum(float(value) for value in penalties.values())
