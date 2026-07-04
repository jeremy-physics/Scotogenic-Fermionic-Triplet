import numpy as np

from constants import nu_sup_exp, nu_sup_inf


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
    return sum(float(value) for value in penalties.values())
