import numpy as np

from constants import (
    GF,
    GammaCaptureAl,
    FpAl,
    GVdn,
    GVdp,
    GVsn,
    GVsp,
    GVun,
    GVup,
    Qd,
    Qs,
    Qu,
    ZAl,
    ZeffAl,
    alpha_e,
    me_exp,
    mmu_exp,
    mtau_exp,
    nAl,
)


PI = np.pi
LEPTON_MASSES = np.array([me_exp, mmu_exp, mtau_exp], dtype=float)
VECTOR_FORM_FACTORS_PROTON = np.array([GVup, GVdp, GVsp], dtype=float)
VECTOR_FORM_FACTORS_NEUTRON = np.array([GVun, GVdn, GVsn], dtype=float)
QUARK_CHARGES = np.array([Qu, Qd, Qs], dtype=float)


def _loop_with_limit(x, limit_value, numerator):
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, np.nan, dtype=float)

    near_one = np.isclose(x, 1.0, rtol=1e-7, atol=1e-10)
    valid = (x > 0.0) & np.isfinite(x) & ~near_one

    result[near_one] = limit_value
    result[valid] = numerator(x[valid])
    return result


def loop_f(x):
    return _loop_with_limit(
        x,
        1.0 / 12.0,
        lambda z: (1.0 - 6.0 * z + 3.0 * z**2 + 2.0 * z**3 - 6.0 * z**2 * np.log(z))
        / (6.0 * (1.0 - z) ** 4),
    )


def loop_g(x):
    return _loop_with_limit(
        x,
        1.0 / 4.0,
        lambda z: (2.0 - 9.0 * z + 18.0 * z**2 - 11.0 * z**3 + 6.0 * z**3 * np.log(z))
        / (6.0 * (1.0 - z) ** 4),
    )


def loop_d1p(x):
    return _loop_with_limit(
        x,
        -1.0 / 3.0,
        lambda z: (-1.0 + z**2 - 2.0 * z * np.log(z)) / (1.0 - z) ** 3,
    )


def loop_d2p(x):
    return _loop_with_limit(
        x,
        1.0 / 6.0,
        lambda z: (-2.0 + 2.0 * z - (1.0 + z) * np.log(z)) / (1.0 - z) ** 3,
    )


def _as_positive_rate(value):
    value = np.real_if_close(value)
    value = float(np.real(value))
    if not np.isfinite(value):
        return np.nan
    return max(value, 0.0)


def _prepare_lfv_inputs(y_xi, fermion_mass, charged_masses):
    y_xi = np.asarray(y_xi, dtype=complex)
    charged_masses = np.asarray(charged_masses, dtype=float).ravel()
    fermion_mass = float(fermion_mass)

    if y_xi.shape != (3, 2):
        raise ValueError(f"LFV Yukawa matrix must have shape (3, 2), got {y_xi.shape}.")
    if charged_masses.size != 2:
        raise ValueError(
            f"LFV charged scalar mass array must contain 2 entries, got {charged_masses.size}."
        )
    if fermion_mass <= 0.0 or not np.isfinite(fermion_mass):
        raise ValueError("LFV fermion mass must be positive and finite.")
    if np.any(charged_masses <= 0.0) or not np.all(np.isfinite(charged_masses)):
        raise ValueError("LFV charged scalar masses must be positive and finite.")

    xi = (fermion_mass / charged_masses) ** 2
    return y_xi, charged_masses, xi


def _idx(index_one_based):
    return int(index_one_based) - 1


def a_non_dipole(i, j, y_xi, charged_masses, xi):
    i0 = _idx(i)
    j0 = _idx(j)
    yukawa_sum = np.sum(y_xi[i0, :] * y_xi[j0, :])
    scalar_sum = np.sum(loop_g(xi) / charged_masses**2)
    return yukawa_sum * scalar_sum / (6.0 * (4.0 * PI) ** 2)


def a_dipole(i, j, y_xi, charged_masses, xi):
    i0 = _idx(i)
    j0 = _idx(j)
    yukawa_sum = np.sum(y_xi[i0, :] * y_xi[j0, :])
    scalar_sum = np.sum(loop_f(xi) / charged_masses**2)
    return yukawa_sum * scalar_sum / (2.0 * (4.0 * PI) ** 2)


def box_b(alpha, beta, y_xi, charged_masses, xi):
    alpha0 = _idx(alpha)
    beta0 = _idx(beta)
    prefactor = 1.0 / ((4.0 * PI) ** 2 * 4.0 * PI * alpha_e)

    total = 0.0j
    d1p = loop_d1p(xi)
    d2p = loop_d2p(xi)
    for scalar_index in range(2):
        y_beta = y_xi[beta0, scalar_index]
        y_alpha = y_xi[alpha0, scalar_index]
        inv_mass_sq = 1.0 / charged_masses[scalar_index] ** 2
        term_d1 = 0.5 * d1p[scalar_index] * np.conj(y_beta) * y_beta * np.conj(y_beta) * y_alpha
        term_d2 = (
            np.sqrt(xi[scalar_index] * xi[scalar_index])
            * d2p[scalar_index]
            * np.conj(y_beta)
            * np.conj(y_beta)
            * y_beta
            * y_alpha
        )
        total += inv_mass_sq * (term_d1 + term_d2)

    return prefactor * total


def branching_gamma(i, j, y_xi, charged_masses, xi):
    ad = a_dipole(i, j, y_xi, charged_masses, xi)
    prefactor = 3.0 * (4.0 * PI) ** 3 * alpha_e / (4.0 * GF**2)
    return _as_positive_rate(prefactor * np.abs(ad) ** 2)


def branching_three_leptons(i, j, y_xi, charged_masses, xi):
    and_amp = a_non_dipole(i, j, y_xi, charged_masses, xi)
    ad = a_dipole(i, j, y_xi, charged_masses, xi)
    b_amp = box_b(i, j, y_xi, charged_masses, xi)
    log_term = np.log(LEPTON_MASSES[_idx(i)] / LEPTON_MASSES[_idx(j)])

    interference = (
        -2.0 * and_amp * np.conj(ad)
        + (1.0 / 3.0) * and_amp * np.conj(b_amp)
        - (2.0 / 3.0) * ad * np.conj(b_amp)
    )
    total = (
        np.abs(and_amp) ** 2
        + np.abs(ad) ** 2 * ((16.0 / 3.0) * log_term - 22.0 / 3.0)
        + (1.0 / 6.0) * np.abs(b_amp) ** 2
        + interference
        + np.conj(interference)
    )
    prefactor = 3.0 * (4.0 * PI) ** 2 * alpha_e**2 / (8.0 * GF**2)
    return _as_positive_rate(prefactor * total)


def _vector_couplings(i, j, y_xi, charged_masses, xi):
    photon_term = a_non_dipole(i, j, y_xi, charged_masses, xi) - a_dipole(
        i, j, y_xi, charged_masses, xi
    )
    return np.sqrt(2.0) / GF * 4.0 * PI * alpha_e * QUARK_CHARGES * photon_term


def _isoscalar_vector(g_vector):
    return 0.5 * np.dot(g_vector, VECTOR_FORM_FACTORS_PROTON + VECTOR_FORM_FACTORS_NEUTRON)


def _isovector_vector(g_vector):
    return 0.5 * np.dot(g_vector, VECTOR_FORM_FACTORS_PROTON - VECTOR_FORM_FACTORS_NEUTRON)


def conversion_rate_aluminum(i, j, y_xi, charged_masses, xi):
    g_lv = _vector_couplings(i, j, y_xi, charged_masses, xi)
    g_rv = g_lv

    g0_lv = _isoscalar_vector(g_lv)
    g1_lv = _isovector_vector(g_lv)
    g0_rv = _isoscalar_vector(g_rv)
    g1_rv = _isovector_vector(g_rv)

    left_amp = (ZAl + nAl) * g0_lv + (ZAl - nAl) * g1_lv
    right_amp = (ZAl + nAl) * g0_rv + (ZAl - nAl) * g1_rv
    prefactor = (
        mmu_exp**5
        * GF**2
        * alpha_e**3
        * ZeffAl**4
        * FpAl**2
        / (8.0 * PI**2 * ZAl * GammaCaptureAl)
    )
    return _as_positive_rate(prefactor * (np.abs(left_amp) ** 2 + np.abs(right_amp) ** 2))


def compute_lfv_observables(y_xi, fermion_mass, charged_masses):
    y_xi, charged_masses, xi = _prepare_lfv_inputs(y_xi, fermion_mass, charged_masses)

    br_mu_e = branching_gamma(2, 1, y_xi, charged_masses, xi)
    br_tau_mu = branching_gamma(3, 2, y_xi, charged_masses, xi)
    br_mu_3e = branching_three_leptons(2, 1, y_xi, charged_masses, xi)
    cr_mu_al_e = conversion_rate_aluminum(2, 1, y_xi, charged_masses, xi)

    return {
        "BR_mu_e": br_mu_e,
        "BR_tau_mu": br_tau_mu,
        "BR_mu_3e": br_mu_3e,
        "CR_muAL_e": cr_mu_al_e,
    }
