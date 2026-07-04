import os
import sys

import numpy as np


# --- Module Imports ---
try:
    from model_definitions import *
    from obs_dark_matter import compute_direct_detection_observables
    from obs_leptons import LNobs
    from obs_scalars import scalar_sector_observables
    from constants import *
    from penalties import compute_restriction_penalty
    from targets import compute_chi_from_targets
    from variables import (
        ALL_MODEL_VARIABLES,
        CASAS_IBARRA_PARAMETERS,
        COMPUTED_SCALAR_COUPLINGS,
        COST_BREAKDOWN,
        DIRECT_DETECTION,
        HIGGS_OBSERVABLES,
        INPUT_QUARTIC_COUPLINGS,
        INPUT_PARAMETERS,
        LAMBDA_MATRIX,
        NEUTRINO_DATA,
        NEUTRINO_MASSES,
        NEUTRINO_YUKAWAS,
        QUADRATIC_PARAMETERS,
        SCALAR_MASSES,
        SCALAR_MIXING,
        TRILINEAR_COUPLINGS,
        get_raw_variables,
        values_to_row,
    )
    from bounds import LOG_IDX, MAX_EXP, MIN_EXP, bounds, get_active_bounds, to_physical

except ImportError as e:
    sys.exit(f"[ERROR] Critical module missing: {e}")


def _active_observables_from_env():
    raw = os.environ.get("ELBAPH_ACTIVE_OBSERVABLES", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


ACTIVE_OBSERVABLES = _active_observables_from_env()
CURRENT_SETTINGS = None


def configure_from_settings(settings):
    global ACTIVE_OBSERVABLES, CURRENT_SETTINGS
    CURRENT_SETTINGS = settings
    ACTIVE_OBSERVABLES = list(getattr(settings, "ACTIVE_OBSERVABLES", []))


def get_effective_settings(settings=None):
    if settings is not None:
        effective_settings = settings
    elif CURRENT_SETTINGS is not None:
        effective_settings = CURRENT_SETTINGS
    else:
        from io_tools import get_settings_file, load_settings, publish_settings_environment

        settings_file = get_settings_file()
        effective_settings = load_settings(settings_file)
        publish_settings_environment(effective_settings)

    if effective_settings is not CURRENT_SETTINGS:
        configure_from_settings(effective_settings)
    return effective_settings


def _empty_values(p_phys):
    values = dict(zip(INPUT_PARAMETERS, p_phys))
    for key in ALL_MODEL_VARIABLES:
        values.setdefault(key, np.nan)
    return values


def _failed_values(p_phys, effective_settings, pen_structure=1e20):
    values = _empty_values(p_phys)
    values["sum_nu"] = np.inf
    values["pen_structure"] = float(pen_structure)
    values["pen_restriction"] = compute_restriction_penalty(values, effective_settings)
    values["penalty_total"] = values["pen_structure"] + values["pen_restriction"]
    values["chi"] = compute_chi_from_targets(values, effective_settings)
    values["cost_function"] = values["chi"] + values["penalty_total"]
    return values


def _lambda_phi_from_higgs_mass(vsigma, lambda_phi_sigma, lambda_sigma):
    numerator = mh_exp**4 - 2.0 * mh_exp**2 * vsigma**2 * lambda_sigma - vphi**2 * vsigma**2 * lambda_phi_sigma**2
    denominator = 2.0 * vphi**2 * (mh_exp**2 - 2.0 * vsigma**2 * lambda_sigma)
    if denominator == 0.0:
        return np.nan
    return numerator / denominator


def _real_scalar(value):
    value = np.real_if_close(value)
    return float(np.real(value))


def _build_values(params, effective_settings):
    p_phys = to_physical(params)

    try:
        (
            A1,
            A2,
            C,
            lambda_eta1_phi,
            lambda_eta1_sigma,
            lambda_12_phi,
            lambda_12_sigma,
            lambda_eta2_phi,
            lambda_eta2_sigma,
            lambda_phi_sigma,
            lambda_sigma,
            mu_12,
            mu_eta1,
            mu_eta2,
            mu_phi,
            zXi,
            z,
        ) = p_phys

        mXi = float(getattr(effective_settings, "FERMION_DM_MASS", 2600.0))
        vsigma = 0.5 * mXi / zXi
        lambda_phi = _lambda_phi_from_higgs_mass(vsigma, lambda_phi_sigma, lambda_sigma)

        if not np.all(np.isfinite([mXi, vsigma, lambda_phi])):
            return _failed_values(p_phys, effective_settings)

        mvis_matrix = MVIS(vphi, vsigma, lambda_phi, lambda_phi_sigma, lambda_sigma)
        me_matrix = ME(
            vphi,
            vsigma,
            A1,
            A2,
            C,
            lambda_eta1_phi,
            lambda_eta1_sigma,
            mu_eta1,
            lambda_12_phi,
            lambda_12_sigma,
            mu_12,
            lambda_eta2_phi,
            lambda_eta2_sigma,
            mu_eta2,
            mu_phi,
        )
        mo_matrix = MO(
            vphi,
            vsigma,
            A1,
            A2,
            C,
            lambda_eta1_phi,
            lambda_eta1_sigma,
            mu_eta1,
            lambda_12_phi,
            lambda_12_sigma,
            mu_12,
            lambda_eta2_phi,
            lambda_eta2_sigma,
            mu_eta2,
            mu_phi,
        )
        mch_matrix = MCH(
            vphi,
            vsigma,
            lambda_eta1_phi,
            lambda_eta1_sigma,
            lambda_12_phi,
            lambda_12_sigma,
            lambda_eta2_phi,
            lambda_eta2_sigma,
            mu_eta1,
            mu_eta2,
            mu_12,
        )

        alpha = 0.5 * np.atan2(
            2.0 * vphi * vsigma * lambda_phi_sigma,
            2.0 * vphi**2 * lambda_phi - 2.0 * vsigma**2 * lambda_sigma,
        )
        theta_c = 0.5 * np.atan2(
            2.0 * vphi**2 * lambda_12_phi + 2.0 * vsigma**2 * lambda_12_sigma + 4.0 * mu_12**2,
            (
                vphi**2 * (lambda_eta1_phi - lambda_eta2_phi)
                + vsigma**2 * (lambda_eta1_sigma - lambda_eta2_sigma)
                + 2.0 * (mu_eta1**2 - mu_eta2**2)
            ),
        )

        gp1 = acople_trilineal_h_CH_CH(
            vphi,
            vsigma,
            theta_c,
            alpha,
            lambda_12_sigma,
            lambda_12_phi,
            lambda_eta1_sigma,
            lambda_eta1_phi,
            lambda_eta2_sigma,
            lambda_eta2_phi,
        )
        gp2 = acople_trilineal_h_CH2_CH2(
            vphi,
            vsigma,
            theta_c,
            alpha,
            lambda_12_sigma,
            lambda_12_phi,
            lambda_eta1_sigma,
            lambda_eta1_phi,
            lambda_eta2_sigma,
            lambda_eta2_phi,
        )

        [mR, mH_vis, mI, mCH, RR, RI, Rgamma], penS = scalar_sector_observables(
            me_matrix,
            mvis_matrix,
            mo_matrix,
            mch_matrix,
            gp1,
            gp2,
            alpha,
        )

        if penS >= 1e20 or not np.all(np.isfinite(np.r_[mR, mH_vis, mI, mCH])):
            return _failed_values(p_phys, effective_settings, pen_structure=penS)

        mh = mH_vis[0]
        mHH = mH_vis[1]
        kappaF = np.cos(alpha)

        Lamb = LAMBDA(mXi, mI, mR, RI, RR)
        U_exp = build_pmns_matrix(s12l_exp, s23l_exp, s13l_exp, deltaCPl_exp)
        mnu_diag_exp = build_mnu_diagonal(dm21_exp, dm31_exp)
        Y = compute_Yukawa_CI(U_exp, mnu_diag_exp, Lamb, z)
        Mneu = build_mnu_active(Y, Lamb)
        Mnsq = Mneu @ Mneu.conj().T

        [m1nu, m2nu, m3nu, dm21, dm31, s13l, s12l, s23l, deltaCP], penLN = LNobs(Mnsq)
        sum_nu = m1nu + m2nu + m3nu

        direct_detection = compute_direct_detection_observables(
            mXi,
            vsigma,
            alpha,
            mh,
            mHH,
            RR,
            RI,
            Y,
            mR,
            mI,
            A1,
            A2,
            C,
            lambda_eta1_phi,
            lambda_eta1_sigma,
            lambda_12_phi,
            lambda_12_sigma,
            lambda_eta2_phi,
            lambda_eta2_sigma,
        )

        values = _empty_values(p_phys)
        values.update(dict(zip(TRILINEAR_COUPLINGS, [A1, A2, C])))
        values.update(
            dict(
                zip(
                    INPUT_QUARTIC_COUPLINGS,
                    [
                        lambda_eta1_phi,
                        lambda_eta1_sigma,
                        lambda_12_phi,
                        lambda_12_sigma,
                        lambda_eta2_phi,
                        lambda_eta2_sigma,
                        lambda_phi_sigma,
                        lambda_sigma,
                    ],
                )
            )
        )
        values.update(dict(zip(QUADRATIC_PARAMETERS, [mu_12, mu_eta1, mu_eta2, mu_phi])))
        values.update(dict(zip(CASAS_IBARRA_PARAMETERS, [zXi, z])))
        values.update(dict(zip(COMPUTED_SCALAR_COUPLINGS, [lambda_phi])))
        values.update(
            dict(
                zip(
                    SCALAR_MASSES,
                    [
                        mh,
                        mHH,
                        mR[0],
                        mR[1],
                        mR[2],
                        mI[0],
                        mI[1],
                        mI[2],
                        mCH[0],
                        mCH[1],
                    ],
                )
            )
        )
        values.update(dict(zip(SCALAR_MIXING, [alpha, theta_c, vsigma, mXi])))
        values.update(dict(zip(HIGGS_OBSERVABLES, [Rgamma, Rgamma, kappaF])))
        values.update(dict(zip(NEUTRINO_DATA, [dm21, dm31, s13l, s12l, s23l, deltaCP])))
        values.update(dict(zip(NEUTRINO_MASSES, [m1nu, m2nu, m3nu, sum_nu])))
        values.update(dict(zip(NEUTRINO_YUKAWAS, np.abs(Y).ravel())))
        values.update(
            dict(
                zip(
                    LAMBDA_MATRIX,
                    [
                        _real_scalar(Lamb[0, 0]),
                        _real_scalar(Lamb[0, 1]),
                        _real_scalar(Lamb[1, 0]),
                        _real_scalar(Lamb[1, 1]),
                    ],
                )
            )
        )
        values.update({key: direct_detection[key] for key in DIRECT_DETECTION})

        pen_structure = penLN + penS
        pen_restriction = compute_restriction_penalty(values, effective_settings)
        penalty_total = pen_structure + pen_restriction
        chi = compute_chi_from_targets(values, effective_settings)
        cost_function = chi + penalty_total

        values.update(
            dict(
                zip(
                    COST_BREAKDOWN,
                    [chi, pen_structure, pen_restriction, penalty_total, cost_function],
                )
            )
        )
        return values

    except Exception:
        return _failed_values(p_phys, effective_settings)


def evaluate_values(point_scan, settings=None):
    effective_settings = get_effective_settings(settings)
    return _build_values(point_scan, effective_settings)


def evaluate_cost_breakdown(point_scan):
    values = evaluate_values(point_scan)

    return {
        "chi": values["chi"],
        "pen_structure": values["pen_structure"],
        "pen_restriction": values["pen_restriction"],
        "penalty_total": values["penalty_total"],
        "cost_function": values["cost_function"],
    }


def chisquare_wrapper(params):
    return evaluate_cost_breakdown(params)["cost_function"]


def evaluate_raw(point_scan, raw_keys=None):
    values = evaluate_values(point_scan)
    labels = list(raw_keys) if raw_keys is not None else get_raw_labels()
    row = values_to_row(values, labels, requested_by="RAW_VARIABLE_SETS")
    physical_array = np.array(to_physical(point_scan), dtype=float)

    return row, values["cost_function"], physical_array, row


def get_raw_labels(settings=None):
    return get_raw_variables(settings or CURRENT_SETTINGS)


def get_raw_header(settings=None):
    return ", ".join(get_raw_labels(settings))
