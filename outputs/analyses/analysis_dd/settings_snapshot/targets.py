import numpy as np

from constants import *


CHI_TARGETS = {
    "dm21": {"model_key": "dm21", "exp": dm21_exp, "err": dm21_err},
    "dm31": {"model_key": "dm31", "exp": dm31_exp, "err": dm31_err},
    "s12l": {"model_key": "s12l", "exp": s12l_exp, "err": s12l_err},
    "s23l": {"model_key": "s23l", "exp": s23l_exp, "err": s23l_err},
    "s13l": {"model_key": "s13l", "exp": s13l_exp, "err": s13l_err},
    "deltaCP": {"model_key": "deltaCP", "exp": deltaCPl_exp, "err": deltaCPl_err},
    "mh": {"model_key": "mh", "exp": mh_exp, "err": mh_err},
    "diph": {"model_key": "diph", "exp": diph_exp, "err": diph_err},
    "kappaF": {"model_key": "kappaF", "exp": kW_exp, "err": kW_err},
    "sigma_si": {"model_key": "sigma_si", "exp": sigma_si_exp, "err": sigma_si_err},
}


CHI_TARGET_SETS = {
    "neutrino_data": [
        "dm21",
        "dm31",
        "s12l",
        "s13l",
        "s23l",
        "deltaCP",
    ],
    "higgs_mass": [
        "mh",
    ],
    "higgs_couplings": [
        "diph",
        "kappaF",
    ],
    "direct_detection_observables": ["sigma_si"],
}


OBSERVABLE_TO_CHI_TARGET_SETS = {
#    "leptons": ["neutrino_data"],
    "neutrinos": ["neutrino_data"],
    "scalars": ["higgs_mass", "higgs_couplings"],
#    "direct_detection": ["direct_detection_observables"],
    "dark_matter": ["direct_detection_observables"],
    "lfv": [],
    "lepton_flavor_violation": [],
}


DEFAULT_ACTIVE_CHI_TARGET_SETS = None
_WARNED_NO_SETTINGS = False


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def get_active_chi_target_sets(settings=None):
    configured_sets = getattr(settings, "ACTIVE_CHI_TARGET_SETS", DEFAULT_ACTIVE_CHI_TARGET_SETS)
    if configured_sets is not None:
        return _dedupe_preserve_order(configured_sets)

    target_sets = []
    active_observables = getattr(settings, "ACTIVE_OBSERVABLES", []) if settings is not None else []
    for observable_name in active_observables or []:
        if observable_name not in OBSERVABLE_TO_CHI_TARGET_SETS:
            print(f"[WARN] Active observable has no chi target mapping: {observable_name}")
            continue
        target_sets.extend(OBSERVABLE_TO_CHI_TARGET_SETS[observable_name])
    return _dedupe_preserve_order(target_sets)


def get_active_chi_targets(settings=None):
    target_sets = get_active_chi_target_sets(settings)
    target_names = []
    for set_name in target_sets or []:
        if set_name not in CHI_TARGET_SETS:
            print(f"[WARN] Chi target set not found: {set_name}")
            continue
        target_names.extend(CHI_TARGET_SETS[set_name])

    resolved_names = []
    for target_name in _dedupe_preserve_order(target_names):
        if target_name not in CHI_TARGETS:
            print(f"[WARN] Chi target not found: {target_name}")
            continue
        resolved_names.append(target_name)
    return resolved_names


def get_chi_target_catalog(settings=None):
    target_catalog = {
        target_name: dict(target)
        for target_name, target in CHI_TARGETS.items()
    }
    custom_targets = getattr(settings, "CUSTOM_CHI_TARGETS", {}) if settings is not None else {}

    if custom_targets is None:
        custom_targets = {}
    if not isinstance(custom_targets, dict):
        print("[warning] CUSTOM_CHI_TARGETS must be a dictionary; overrides ignored.")
        return target_catalog

    for target_name, overrides in custom_targets.items():
        if target_name not in target_catalog:
            print(f"[warning] CUSTOM_CHI_TARGETS contains unknown target '{target_name}'.")
            continue
        if not isinstance(overrides, dict):
            print(
                f"[warning] CUSTOM_CHI_TARGETS entry '{target_name}' must be a dictionary; "
                "override ignored."
            )
            continue

        for field_name, value in overrides.items():
            if field_name not in target_catalog[target_name]:
                print(
                    f"[warning] CUSTOM_CHI_TARGETS target '{target_name}' contains "
                    f"unknown field '{field_name}'."
                )
                continue
            target_catalog[target_name][field_name] = value

    return target_catalog


def get_chi_targets(settings=None):
    target_catalog = get_chi_target_catalog(settings)
    resolved_targets = []
    for target_name in get_active_chi_targets(settings):
        target = dict(target_catalog[target_name])
        target["name"] = target_name
        resolved_targets.append(target)
    return resolved_targets


def get_active_chi_model_keys(settings=None):
    model_keys = []
    seen = set()
    for target in get_chi_targets(settings):
        model_key = target["model_key"]
        if model_key in seen:
            continue
        seen.add(model_key)
        model_keys.append(model_key)
    return model_keys


def compute_chi_from_targets(values, settings=None):
    global _WARNED_NO_SETTINGS
    if settings is None and not _WARNED_NO_SETTINGS:
        print(
            "[WARN] compute_chi_from_targets called with settings=None. "
            "No chi targets selected."
        )
        _WARNED_NO_SETTINGS = True

    sigma_scale = float(getattr(settings, "CHI_SIGMA_SCALE", 3.0))
    chi = 0.0

    for target in get_chi_targets(settings):
        target_name = target["name"]
        model_key = target["model_key"]
        if model_key not in values:
            print(
                f"[WARN] Chi target '{target_name}' model_key not found in values: "
                f"{model_key}"
            )
            continue

        model_value = values[model_key]
        exp_value = target["exp"]
        err_value = sigma_scale * target["err"]
        if err_value == 0:
            print(f"[WARN] Chi target has zero error and was skipped: {target_name}")
            continue

        contribution = ((model_value - exp_value) / err_value) ** 2
        if np.isfinite(contribution):
            chi += contribution
        else:
            chi += 1e20

    return float(chi)
