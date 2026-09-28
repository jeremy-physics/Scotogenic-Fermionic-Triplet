import numpy as np

from variables import INPUT_PARAMETERS


pi = np.pi

trilinear_lat = (-1.0, 1.0)
quartic_lat = (-1.0, 1.0)
quadratic_lat = (-1.0, 1.0)
z_lat = (0.0, 1.0)
phase = (0.0, 2.0 * pi)

bounds = (
    [trilinear_lat] * 3
    + [quartic_lat] * 8
    + [quadratic_lat] * 4
    + [z_lat]
    + [phase]
)

LOG_IDX = np.arange(16)

MIN_EXP = np.array([2] * 3 + [-4] * 8 + [2] * 4 + [-5])
MAX_EXP = np.array([4] * 3 + [0] * 8 + [4] * 4 + [-1])#zXi up to log10(5e-2)


SCAN_PARAMETER_NAMES = list(INPUT_PARAMETERS)


def to_physical(params):
    p_phys = np.array(params, dtype=float)
    x = p_phys[LOG_IDX]

    signs = np.where(x >= 0, 1.0, -1.0)
    exponents = MIN_EXP + np.abs(x) * (MAX_EXP - MIN_EXP)

    p_phys[LOG_IDX] = signs * (10.0 ** exponents)
    return p_phys


def get_active_bounds(settings=None):
    active_bounds = [tuple(bound) for bound in bounds]
    custom_bounds = getattr(settings, "CUSTOM_BOUNDS", {}) if settings is not None else {}

    for parameter_name, custom_bound in (custom_bounds or {}).items():
        if parameter_name not in SCAN_PARAMETER_NAMES:
            print(
                f"[WARN] CUSTOM_BOUNDS parameter not found in "
                f"SCAN_PARAMETER_NAMES: {parameter_name}"
            )
            continue

        try:
            low, high = custom_bound
        except (TypeError, ValueError):
            print(
                f"[WARN] Invalid CUSTOM_BOUNDS value for {parameter_name}: "
                f"{custom_bound}. Expected (low, high)."
            )
            continue

        index = SCAN_PARAMETER_NAMES.index(parameter_name)
        active_bounds[index] = (low, high)

    return active_bounds
