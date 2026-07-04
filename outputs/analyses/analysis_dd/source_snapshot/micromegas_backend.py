from ctypes import CDLL, POINTER, byref, c_double, c_int
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
N_MICROMEGAS_INPUTS = 18


def resolve_library_path(library_path):
    path = Path(library_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _settings_value(settings, name, default=None):
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _error(status, message):
    return {
        "Omega": None,
        "sigmaSI": None,
        "status": status,
        "status_code": None,
        "message": message,
    }


def _with_library_cwd(lib_path, callback):
    original_cwd = os.getcwd()
    try:
        os.chdir(lib_path.parent)
        return callback()
    finally:
        os.chdir(original_cwd)


def _run_status_api(library, lib_path, inputs, function_name):
    try:
        runner = getattr(library, function_name)
    except AttributeError:
        return None

    values = (c_double * len(inputs))(*[float(value) for value in inputs])
    omega = c_double()
    sigma_si = c_double()

    runner.argtypes = [
        POINTER(c_double),
        c_int,
        POINTER(c_double),
        POINTER(c_double),
    ]
    runner.restype = c_int

    try:
        status_code = _with_library_cwd(
            lib_path,
            lambda: runner(values, c_int(len(inputs)), byref(omega), byref(sigma_si)),
        )
    except Exception as e:
        return _error("error", f"MicrOMEGAs call failed: {e}")

    if status_code != 0:
        return {
            "Omega": None,
            "sigmaSI": None,
            "status": "error",
            "status_code": int(status_code),
            "message": f"MicrOMEGAs returned status {int(status_code)}.",
        }

    return {
        "Omega": float(omega.value),
        "sigmaSI": float(sigma_si.value),
        "status": "ok",
        "status_code": 0,
        "message": "ok",
        "backend_api": function_name,
    }


def _run_legacy_api(library, lib_path, inputs):
    initialize = getattr(library, "initialize_micromegas", None)
    get_observables = getattr(library, "get_observables", None)
    if get_observables is None:
        return None

    if initialize is not None:
        initialize.argtypes = []
        initialize.restype = None

    get_observables.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
    ]
    get_observables.restype = None

    values = (c_double * len(inputs))(*[float(value) for value in inputs])
    results = (c_double * 2)()

    def run_call():
        if initialize is not None:
            initialize()
        get_observables(values, results)

    try:
        _with_library_cwd(lib_path, run_call)
    except Exception as e:
        return _error("error", f"Legacy MicrOMEGAs call failed: {e}")

    omega = float(results[0])
    sigma_si = float(results[1])
    if omega < 0:
        return {
            "Omega": None,
            "sigmaSI": None,
            "status": "error",
            "status_code": 12,
            "message": "Legacy MicrOMEGAs returned invalid Omega sentinel.",
        }

    return {
        "Omega": omega,
        "sigmaSI": sigma_si,
        "status": "ok",
        "status_code": 0,
        "message": "ok",
        "backend_api": "legacy_get_observables",
    }


def run_micromegas(inputs, settings=None, library_path=None, function_name="run_micromegas"):
    if isinstance(settings, (str, Path)) and library_path is None:
        library_path = settings
        settings = None

    if len(inputs) != N_MICROMEGAS_INPUTS:
        return _error(
            "error",
            f"MicrOMEGAs input size mismatch: expected {N_MICROMEGAS_INPUTS}, got {len(inputs)}.",
        )

    if library_path is None:
        library_path = _settings_value(settings, "MICROMEGAS_LIB")

    if library_path is None:
        library_path = os.environ.get("ELBAPH_MICROMEGAS_LIB")

    if not library_path:
        return _error(
            "missing_settings",
            "dark_matter is active but MICROMEGAS_LIB is not defined in settings.",
        )

    lib_path = resolve_library_path(library_path)
    if not lib_path.exists():
        return _error("missing_library", f"MicrOMEGAs library not found: {lib_path}")

    try:
        library = CDLL(str(lib_path))
    except OSError as e:
        return _error("load_error", f"Could not load MicrOMEGAs library {lib_path}: {e}")

    result = _run_status_api(library, lib_path, inputs, function_name)
    if result is not None:
        return result

    result = _run_legacy_api(library, lib_path, inputs)
    if result is not None:
        return result

    return _error(
        "missing_symbol",
        (
            f"Neither '{function_name}' nor legacy 'get_observables' "
            f"was found in {lib_path}."
        ),
    )
