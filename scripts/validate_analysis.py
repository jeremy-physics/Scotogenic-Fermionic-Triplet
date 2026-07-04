import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np


# === USER/AGENT CONFIGURATION START ===

USE_CURRENT_ANALYSIS = True
LOCAL_SETTINGS_FILE = None

# === USER/AGENT CONFIGURATION END ===


CURRENT_DIR = Path(os.path.abspath(__file__)).parent
PROJECT_ROOT = CURRENT_DIR.parent
PROJECT_SRC = PROJECT_ROOT / "src"

module_path = str(PROJECT_SRC)
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from bounds import SCAN_PARAMETER_NAMES, bounds, get_active_bounds
from io_tools import get_analysis_dir, get_settings_file, load_settings, resolve_project_path
from likelihood import configure_from_settings, evaluate_values
from targets import (
    CHI_TARGETS,
    CHI_TARGET_SETS,
    DEFAULT_ACTIVE_CHI_TARGET_SETS,
    OBSERVABLE_TO_CHI_TARGET_SETS,
    get_active_chi_target_sets,
    get_chi_targets,
)
from variables import (
    DEFAULT_RAW_VARIABLE_SETS,
    DEFAULT_SHOW_VARIABLE_SETS,
    DYNAMIC_VARIABLE_SETS,
    VARIABLE_SETS,
    get_raw_variables,
    get_show_variables,
)


SECTION_NAMES = (
    "Settings",
    "Bounds",
    "Chi targets",
    "Raw/show variables",
    "micrOMEGAs",
    "Benchmark evaluation",
)


def new_results():
    return {
        section: {"state": "OK", "warnings": 0}
        for section in SECTION_NAMES
    }


def add_warning(results, section, message):
    results[section]["warnings"] += 1
    results[section]["state"] = "warnings"
    print(f"[warning] {message}")


def set_skipped(results, section):
    if results[section]["warnings"] == 0:
        results[section]["state"] = "skipped"


def call_with_warnings(results, section, function, fallback, report_messages=True):
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            value = function()
    except Exception as exc:
        add_warning(results, section, str(exc))
        return fallback

    for line in stream.getvalue().splitlines():
        stripped = line.strip()
        if not stripped or not report_messages:
            continue
        results[section]["warnings"] += 1
        results[section]["state"] = "warnings"
        print(stripped)
    return value


def get_optional_setting(results, settings, name, default):
    if hasattr(settings, name):
        return getattr(settings, name)
    add_warning(results, "Settings", f"Optional setting {name} is missing; using {default!r}.")
    return default


def validate_settings(results, settings):
    analysis_name = getattr(settings, "ANALYSIS_NAME", None)
    if not analysis_name:
        add_warning(results, "Settings", "ANALYSIS_NAME is missing or empty.")

    default_chi_target_sets = (
        None
        if DEFAULT_ACTIVE_CHI_TARGET_SETS is None
        else list(DEFAULT_ACTIVE_CHI_TARGET_SETS)
    )
    values = {
        "ANALYSIS_NAME": analysis_name,
        "ACTIVE_OBSERVABLES": get_optional_setting(results, settings, "ACTIVE_OBSERVABLES", []),
        "ACTIVE_CHI_TARGET_SETS": get_optional_setting(
            results,
            settings,
            "ACTIVE_CHI_TARGET_SETS",
            default_chi_target_sets,
        ),
        "RAW_VARIABLE_SETS": get_optional_setting(
            results,
            settings,
            "RAW_VARIABLE_SETS",
            list(DEFAULT_RAW_VARIABLE_SETS),
        ),
        "SHOW_VARIABLE_SETS": get_optional_setting(
            results,
            settings,
            "SHOW_VARIABLE_SETS",
            list(DEFAULT_SHOW_VARIABLE_SETS),
        ),
        "CUSTOM_BOUNDS": get_optional_setting(results, settings, "CUSTOM_BOUNDS", {}),
        "CUSTOM_CHI_TARGETS": get_optional_setting(results, settings, "CUSTOM_CHI_TARGETS", {}),
    }

    print("\nSETTINGS")
    for name, value in values.items():
        print(f"- {name}: {value!r}")
    return values


def validate_bounds(results, settings, settings_values):
    active_bounds = call_with_warnings(
        results,
        "Bounds",
        lambda: get_active_bounds(settings),
        list(bounds),
        report_messages=False,
    )

    print("\nBOUNDS")
    print(f"- active bounds: {len(active_bounds)}")
    print(f"- scan parameter names: {len(SCAN_PARAMETER_NAMES)}")
    if len(active_bounds) != len(SCAN_PARAMETER_NAMES):
        add_warning(
            results,
            "Bounds",
            f"Bounds length {len(active_bounds)} does not match "
            f"SCAN_PARAMETER_NAMES length {len(SCAN_PARAMETER_NAMES)}.",
        )

    custom_bounds = settings_values["CUSTOM_BOUNDS"]
    if not isinstance(custom_bounds, dict):
        add_warning(results, "Bounds", "CUSTOM_BOUNDS must be a dictionary.")
    else:
        for parameter_name, custom_bound in custom_bounds.items():
            if parameter_name not in SCAN_PARAMETER_NAMES:
                add_warning(
                    results,
                    "Bounds",
                    f"CUSTOM_BOUNDS contains unknown parameter '{parameter_name}'.",
                )
                continue
            try:
                low, high = custom_bound
            except (TypeError, ValueError):
                add_warning(
                    results,
                    "Bounds",
                    f"CUSTOM_BOUNDS value for '{parameter_name}' must be (low, high).",
                )

    return active_bounds


def validate_chi_targets(results, settings, settings_values):
    active_targets = call_with_warnings(
        results,
        "Chi targets",
        lambda: get_chi_targets(settings),
        [],
        report_messages=False,
    )

    print("\nCHI TARGETS")
    resolved_sets = call_with_warnings(
        results,
        "Chi targets",
        lambda: get_active_chi_target_sets(settings),
        [],
        report_messages=False,
    )
    print(f"- active sets: {resolved_sets}")
    print(f"- resolved targets: {[target.get('name') for target in active_targets]}")
    required_fields = ("model_key", "exp", "err")
    for target in active_targets:
        target_name = target.get("name", "<unnamed>")
        if target_name not in CHI_TARGETS:
            add_warning(results, "Chi targets", f"Resolved target '{target_name}' is not in CHI_TARGETS.")
        for field_name in required_fields:
            if field_name not in target:
                add_warning(
                    results,
                    "Chi targets",
                    f"Target '{target_name}' is missing required field '{field_name}'.",
                )

    active_sets = settings_values["ACTIVE_CHI_TARGET_SETS"]
    if active_sets is None:
        for observable_name in settings_values["ACTIVE_OBSERVABLES"] or []:
            if observable_name not in OBSERVABLE_TO_CHI_TARGET_SETS:
                add_warning(
                    results,
                    "Chi targets",
                    f"Active observable '{observable_name}' has no chi target mapping.",
                )
        active_sets = resolved_sets
    elif not isinstance(active_sets, (list, tuple)):
        add_warning(results, "Chi targets", "ACTIVE_CHI_TARGET_SETS must be a list or tuple.")
        active_sets = []
    for set_name in active_sets:
        if set_name not in CHI_TARGET_SETS:
            add_warning(results, "Chi targets", f"Unknown chi target set '{set_name}'.")
            continue
        for target_name in CHI_TARGET_SETS[set_name]:
            if target_name not in CHI_TARGETS:
                add_warning(
                    results,
                    "Chi targets",
                    f"Set '{set_name}' contains unknown target '{target_name}'.",
                )

    custom_targets = settings_values["CUSTOM_CHI_TARGETS"]
    if not isinstance(custom_targets, dict):
        add_warning(results, "Chi targets", "CUSTOM_CHI_TARGETS must be a dictionary.")
    else:
        for target_name, overrides in custom_targets.items():
            if target_name not in CHI_TARGETS:
                add_warning(results, "Chi targets", f"Unknown custom chi target '{target_name}'.")
                continue
            if not isinstance(overrides, dict):
                add_warning(
                    results,
                    "Chi targets",
                    f"CUSTOM_CHI_TARGETS entry '{target_name}' must be a dictionary.",
                )
                continue
            for field_name in overrides:
                if field_name not in CHI_TARGETS[target_name]:
                    add_warning(
                        results,
                        "Chi targets",
                        f"CUSTOM_CHI_TARGETS target '{target_name}' has unknown field '{field_name}'.",
                    )

    return active_targets


def validate_raw_show_variables(results, settings, settings_values):
    raw_keys = call_with_warnings(
        results,
        "Raw/show variables",
        lambda: get_raw_variables(settings),
        [],
        report_messages=False,
    )
    show_keys = call_with_warnings(
        results,
        "Raw/show variables",
        lambda: get_show_variables(settings),
        [],
        report_messages=False,
    )

    print("\nRAW/SHOW VARIABLES")
    print(f"- raw_keys: {raw_keys}")
    print(f"- show_keys: {show_keys}")

    for setting_name in ("RAW_VARIABLE_SETS", "SHOW_VARIABLE_SETS"):
        requested_sets = settings_values[setting_name]
        if not isinstance(requested_sets, (list, tuple)):
            add_warning(results, "Raw/show variables", f"{setting_name} must be a list or tuple.")
            continue
        for set_name in requested_sets:
            if set_name not in VARIABLE_SETS and set_name not in DYNAMIC_VARIABLE_SETS:
                add_warning(
                    results,
                    "Raw/show variables",
                    f"{setting_name} contains unknown variable set '{set_name}'.",
                )
    return raw_keys, show_keys


def validate_micromegas(results, settings, settings_values):
    active_observables = settings_values["ACTIVE_OBSERVABLES"]
    print("\nMICROMEGAS")
    if "dark_matter" not in active_observables:
        print("- skipped: dark_matter is not active")
        set_skipped(results, "micrOMEGAs")
        return

    library = getattr(settings, "MICROMEGAS_LIB", None)
    if not library:
        add_warning(results, "micrOMEGAs", "dark_matter is active but MICROMEGAS_LIB is missing.")
        return

    library_path = resolve_project_path(library)
    print(f"- library: {library_path}")
    if not library_path.is_file():
        add_warning(results, "micrOMEGAs", f"MICROMEGAS_LIB does not exist: {library_path}")


def validate_benchmark(results, settings, active_bounds, raw_keys, show_keys):
    print("\nBENCHMARK EVALUATION")
    if not getattr(settings, "ANALYSIS_NAME", None) or not getattr(settings, "OUTPUT_ROOT", None):
        add_warning(results, "Benchmark evaluation", "Cannot resolve Analysis output folder.")
        return

    optimization_dir = get_analysis_dir(settings) / "optimization"
    candidates = (
        optimization_dir / "point_01_R.dat",
        optimization_dir / "point_01.dat",
    )
    point_path = next((path for path in candidates if path.is_file()), None)
    if point_path is None:
        print("No benchmark point found; skipped evaluation test.")
        set_skipped(results, "Benchmark evaluation")
        return

    print(f"- point: {point_path}")
    try:
        point_data = np.loadtxt(point_path, comments="#")
        point_scan = point_data[-1] if np.ndim(point_data) > 1 else point_data
        point_scan = np.asarray(point_scan, dtype=float).ravel()
    except Exception as exc:
        add_warning(results, "Benchmark evaluation", f"Could not read benchmark: {exc}")
        return

    if point_scan.size != len(active_bounds):
        add_warning(
            results,
            "Benchmark evaluation",
            f"Benchmark has {point_scan.size} parameters; expected {len(active_bounds)}.",
        )
        return

    try:
        configure_from_settings(settings)
        values = evaluate_values(point_scan, settings=settings)
    except Exception as exc:
        add_warning(results, "Benchmark evaluation", f"Evaluation failed: {exc}")
        return

    required_cost_keys = (
        "chi",
        "pen_structure",
        "pen_restriction",
        "penalty_total",
        "cost_function",
    )
    for key in required_cost_keys:
        if key not in values:
            add_warning(results, "Benchmark evaluation", f"values is missing required key '{key}'.")

    if all(key in values for key in required_cost_keys):
        expected_penalty = values["pen_structure"] + values["pen_restriction"]
        expected_cost = values["chi"] + values["penalty_total"]
        if not np.isclose(values["penalty_total"], expected_penalty, rtol=1e-12, atol=1e-12):
            add_warning(results, "Benchmark evaluation", "penalty_total identity is not satisfied.")
        if not np.isclose(values["cost_function"], expected_cost, rtol=1e-12, atol=1e-12):
            add_warning(results, "Benchmark evaluation", "cost_function identity is not satisfied.")

    for key in raw_keys:
        if key not in values:
            add_warning(results, "Benchmark evaluation", f"raw key '{key}' is missing from values.")
    for key in show_keys:
        if key not in values:
            add_warning(results, "Benchmark evaluation", f"show key '{key}' is missing from values.")

    if results["Benchmark evaluation"]["warnings"] == 0:
        print(f"- cost_function: {values['cost_function']:.12g}")
        print("- identities: OK")


def print_summary(results):
    print("\nVALIDATION SUMMARY")
    for section in SECTION_NAMES:
        print(f"- {section}: {results[section]['state']}")


def main():
    results = new_results()
    try:
        settings_file = get_settings_file(USE_CURRENT_ANALYSIS, LOCAL_SETTINGS_FILE)
        settings = load_settings(settings_file)
    except Exception as exc:
        add_warning(results, "Settings", f"Could not load settings: {exc}")
        for section in SECTION_NAMES[1:]:
            set_skipped(results, section)
        print_summary(results)
        return

    print(f"SETTINGS FILE: {settings_file}")
    settings_values = validate_settings(results, settings)
    active_bounds = validate_bounds(results, settings, settings_values)
    validate_chi_targets(results, settings, settings_values)
    raw_keys, show_keys = validate_raw_show_variables(results, settings, settings_values)
    validate_micromegas(results, settings, settings_values)
    validate_benchmark(results, settings, active_bounds, raw_keys, show_keys)
    print_summary(results)


if __name__ == "__main__":
    main()
