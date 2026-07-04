import os
import sys
from pathlib import Path

import numpy as np


# === USER/AGENT CONFIGURATION START ===

USE_CURRENT_ANALYSIS = True
LOCAL_SETTINGS_FILE = None

TARGET_FILE = None
POINT_CANDIDATES = [
    "point_01.dat",
    #"bench_02.dat",
    #"test_point_01.dat",
]

PRINT_SCAN_POINT = False
PRINT_PHYSICAL_INPUTS = False
PRINT_SHOW_VARIABLES = True
PRINT_COST_BREAKDOWN = True
PRINT_RELATIVE_ERROR = True
SORT_RELATIVE_ERROR = True
PRINT_RELATIVE_ERROR_DETAILS = False
PRINT_MATH = False
PRINT_MICRO = False

MATH_VARIABLE_SETS = None
EXTRA_MATH_VARIABLES = []

# === USER/AGENT CONFIGURATION END ===


CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = CURRENT_DIR.parent
PROJECT_SRC = PROJECT_ROOT / "src"

module_path = str(PROJECT_SRC)
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from bounds import to_physical
from io_tools import (
    configure_loaded_likelihood,
    get_analysis_dir,
    get_settings_file,
    load_settings,
    publish_settings_environment,
)
from likelihood import evaluate_values
from targets import get_chi_targets
from variables import (
    INPUT_PARAMETERS,
    get_show_variables,
    get_variables_from_sets,
)


COST_KEYS = (
    "chi",
    "pen_structure",
    "pen_restriction",
    "penalty_total",
    "cost_function",
)

MICRO_VARIABLE_MAP = (
    ("MR", "mR"),
    ("MN1", "MN1"),
    ("MN2", "MN2"),
    ("MN3", "MN3"),
    ("la2", "lambda2"),
    ("la3", "lambda3"),
    ("la4", "lambda4"),
    ("la5", "lambda5"),
    ("YN11", "yN11"),
    ("YN12r", "yN12_re"),
    ("YN12i", "yN12_im"),
    ("YN13", "yN13"),
    ("YN21", "yN21"),
    ("YN22", "yN22"),
    ("YN23", "yN23"),
    ("YN31", "yN31"),
    ("YN32", "yN32"),
    ("YN33", "yN33"),
)


def configure_analysis():
    settings_file = get_settings_file(USE_CURRENT_ANALYSIS, LOCAL_SETTINGS_FILE)
    settings = load_settings(settings_file)
    publish_settings_environment(settings)
    configure_loaded_likelihood(settings)
    optimization_dir = get_analysis_dir(settings) / "optimization"
    return settings, optimization_dir


def select_point_file(optimization_dir):
    if TARGET_FILE is not None:
        target_path = optimization_dir / TARGET_FILE
        if target_path.exists():
            return target_path
        print(f"[WARN] Point file not found: {target_path}")
        return None

    for candidate in POINT_CANDIDATES:
        candidate_path = optimization_dir / candidate
        if candidate_path.exists():
            return candidate_path

    print("[WARN] No benchmark point found in optimization/.")
    print(f"[WARN] Searched in: {optimization_dir}")
    print(f"[WARN] Candidates: {', '.join(POINT_CANDIDATES)}")
    return None


def load_point_scan(point_file):
    try:
        loaded = np.loadtxt(point_file, comments="#")
    except Exception as exc:
        print(f"[ERROR] Could not read point file {point_file}: {exc}")
        return None

    if np.asarray(loaded).ndim > 1:
        print(f"[WARN] Point file contains multiple rows; using the last row: {point_file}")
        loaded = np.asarray(loaded)[-1]

    point_scan = np.asarray(loaded, dtype=float).ravel()
    if not np.all(np.isfinite(point_scan)):
        print(f"[ERROR] Point file contains non-finite values: {point_file}")
        return None
    return point_scan


def print_value(name, value):
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        array = np.asarray(value)
        formatted = np.array2string(
            array,
            formatter={"float_kind": lambda item: f"{item:.6e}"},
            separator=", ",
        )
        print(f"{name} = {formatted}")
        return

    if isinstance(value, (float, np.floating)):
        print(f"{name} = {value:.12e}")
        return

    print(f"{name} = {value}")


def print_physical_parameters(point_scan):
    point_physical = np.asarray(to_physical(point_scan), dtype=float).ravel()
    if point_physical.size != len(INPUT_PARAMETERS):
        print(
            f"[WARN] Physical point has {point_physical.size} values but "
            f"INPUT_PARAMETERS has {len(INPUT_PARAMETERS)} names."
        )
        print_value("point_physical", point_physical)
        return

    for name, value in zip(INPUT_PARAMETERS, point_physical):
        print_value(name, value)


def print_relative_errors(values, settings):
    entries = []
    for target in get_chi_targets(settings):
        target_name = target["name"]
        model_key = target["model_key"]

        if model_key not in values:
            entries.append({
                "name": target_name,
                "missing": True,
                "rel_err": np.nan,
            })
            continue

        model_value = values[model_key]
        exp_value = target["exp"]
        err_value = target["err"]
        relative_error = np.nan if exp_value == 0 else (
            model_value - exp_value
        ) / exp_value
        pull = np.nan if err_value == 0 else (
            model_value - exp_value
        ) / err_value

        entries.append({
            "name": target_name,
            "missing": False,
            "model": model_value,
            "exp": exp_value,
            "err": err_value,
            "rel_err": relative_error,
            "pull": pull,
        })

    if SORT_RELATIVE_ERROR:
        entries.sort(
            key=lambda item: (
                np.isnan(item["rel_err"]),
                -abs(item["rel_err"]) if not np.isnan(item["rel_err"]) else 0.0,
            )
        )

    name_width = max((len(item["name"]) for item in entries), default=1)
    for item in entries:
        target_name = item["name"]
        if item["missing"]:
            print(f"{target_name:<{name_width}} : missing model_key")
            continue

        if not PRINT_RELATIVE_ERROR_DETAILS:
            print(f"{target_name:<{name_width}} : {item['rel_err']:.6e}")
            continue

        print(f"{target_name}:")
        print(f"    model = {item['model']:.12e}")
        print(f"    exp = {item['exp']:.12e}")
        print(f"    err = {item['err']:.12e}")
        print(f"    rel_err = {item['rel_err']:.12e}")
        print(f"    pull = {item['pull']:.12e}")


def scalar_value(value):
    array = np.asarray(value)
    if array.ndim != 0:
        return None
    return array.item()


def format_mathematica_scalar(value):
    scalar = scalar_value(value)
    if scalar is None:
        return None

    if isinstance(scalar, complex):
        if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
            return "Indeterminate"
        return f"({scalar.real:.16e} + {scalar.imag:.16e} I)"

    try:
        numeric = float(scalar)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric):
        return "Indeterminate"
    return f"{numeric:.16e}"


def get_math_variables(show_keys, settings):
    if MATH_VARIABLE_SETS is None:
        return get_variables_from_sets(
            [],
            list(show_keys) + list(EXTRA_MATH_VARIABLES),
            settings=settings,
        )
    return get_variables_from_sets(
        MATH_VARIABLE_SETS,
        EXTRA_MATH_VARIABLES,
        settings=settings,
    )


def print_mathematica_export(values, show_keys, settings):
    rules = []
    for key in get_math_variables(show_keys, settings):
        if key not in values:
            print(f"[WARN] Mathematica variable not found in values: {key}")
            continue

        formatted = format_mathematica_scalar(values[key])
        if formatted is None:
            print(f"[WARN] Mathematica export skipped non-scalar variable: {key}")
            continue
        rules.append(f"{key} -> {formatted}")

    print("{" + ", ".join(rules) + "}")


def print_micromegas_export(values):
    for micro_name, value_key in MICRO_VARIABLE_MAP:
        if value_key not in values:
            print(f"[WARN] micrOMEGAs variable not found in values: {value_key}")
            continue

        scalar = scalar_value(values[value_key])
        try:
            numeric = float(scalar)
        except (TypeError, ValueError):
            print(f"[WARN] micrOMEGAs export skipped non-scalar variable: {value_key}")
            continue

        print(f"{micro_name} {numeric:.20E}")


def print_cost_verification(values):
    required = set(COST_KEYS)
    missing = sorted(required.difference(values))
    if missing:
        print(f"[WARN] Cost verification skipped; missing keys: {', '.join(missing)}")
        return

    penalty_residual = values["penalty_total"] - (
        values["pen_structure"] + values["pen_restriction"]
    )
    cost_residual = values["cost_function"] - (
        values["chi"] + values["penalty_total"]
    )

    penalty_ok = np.isclose(penalty_residual, 0.0, rtol=1e-12, atol=1e-12)
    cost_ok = np.isclose(cost_residual, 0.0, rtol=1e-12, atol=1e-12)

    penalty_status = "OK" if penalty_ok else "MISMATCH"
    cost_status = "OK" if cost_ok else "MISMATCH"
    print(
        "penalty_total = pen_structure + pen_restriction: "
        f"{penalty_status} (residual={penalty_residual:.6e})"
    )
    print(
        "cost_function = chi + penalty_total: "
        f"{cost_status} (residual={cost_residual:.6e})"
    )


def main():
    settings, optimization_dir = configure_analysis()
    point_file = select_point_file(optimization_dir)
    if point_file is None:
        return

    point_scan = load_point_scan(point_file)
    if point_scan is None:
        return

    if point_scan.size != len(INPUT_PARAMETERS):
        print(
            f"[ERROR] Scan-space point has {point_scan.size} values; "
            f"expected {len(INPUT_PARAMETERS)}."
        )
        return

    try:
        values = evaluate_values(point_scan, settings=settings)
    except Exception as exc:
        print(f"[ERROR] Could not evaluate point: {exc}")
        return

    show_keys = get_show_variables(settings)

    print("\nANALYSIS")
    print(settings.ANALYSIS_NAME)

    print("\nPOINT FILE")
    print(point_file)

    if PRINT_SCAN_POINT:
        print("\nSCAN-SPACE POINT")
        print_value("point_scan", point_scan)

    if PRINT_PHYSICAL_INPUTS:
        print("\nPHYSICAL INPUT PARAMETERS")
        print_physical_parameters(point_scan)

    if PRINT_SHOW_VARIABLES:
        print("\nCONFIGURED SHOW VARIABLES")
        for key in show_keys:
            if key in COST_KEYS:
                continue
            if key in values:
                print_value(key, values[key])
            else:
                print(
                    f"[warning] variable '{key}' requested in SHOW_VARIABLE_SETS "
                    "but not found in values."
                )

    if PRINT_RELATIVE_ERROR:
        print("\nRELATIVE ERROR")
        print_relative_errors(values, settings)

    if PRINT_COST_BREAKDOWN:
        print("\nCOST BREAKDOWN")
        for key in COST_KEYS:
            if key in values:
                print_value(key, values[key])
            else:
                print(f"[warning] cost variable '{key}' not found in values.")

    print("\nCONSISTENCY CHECK")
    print_cost_verification(values)

    if PRINT_MATH:
        print("\nMATHEMATICA EXPORT")
        print_mathematica_export(values, show_keys, settings)

    if PRINT_MICRO:
        print("\nMICROMEGAS EXPORT")
        print_micromegas_export(values)


if __name__ == "__main__":
    main()
