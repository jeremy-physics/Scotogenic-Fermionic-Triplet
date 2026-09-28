import os
import sys
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy.stats import qmc


# === USER/AGENT CONFIGURATION START ===

USE_CURRENT_ANALYSIS = True
LOCAL_SETTINGS_FILE = None
DATA_TAG = "scans"
OUTPUT_FILE = "scan_lfv.dat"

# --- USER SETTINGS ---
PENALTY_MAX = 8.0
CHUNK_SIZE = 20000
TARGET_POINTS = 100000
MAX_EVALUATIONS = 100000000
OVERWRITE_OUTPUT = True
LABELS_HEADER = None

# === USER/AGENT CONFIGURATION END ===

# --- Path Configuration ---
CURRENT_DIR = Path(os.path.abspath(__file__)).parenrt
PROJECT_ROOT = CURRENT_DIR.parent
PROJECT_SRC = PROJECT_ROOT / "src"

module_path = str(PROJECT_SRC)
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from io_tools import (
    configure_loaded_likelihood,
    get_settings_file,
    load_settings,
    prepare_analysis,
    publish_settings_environment,
)

# Importamos la fisica
from likelihood import bounds, evaluate_raw
from bounds import get_active_bounds
from variables import get_raw_variables, validate_row_width


# === CONFIGURATION ===
SAVE_DIR = None
OUTPUT_PATH = None
RAW_KEYS = None
ACTIVE_BOUNDS = bounds
WORKER_SETTINGS_FILE = None


def init_worker(settings_file):
    settings = load_settings(settings_file)
    publish_settings_environment(settings)
    configure_loaded_likelihood(settings)


def configure_analysis():
    global ACTIVE_BOUNDS, LABELS_HEADER, OUTPUT_PATH, RAW_KEYS, SAVE_DIR, WORKER_SETTINGS_FILE

    settings_file = get_settings_file(USE_CURRENT_ANALYSIS, LOCAL_SETTINGS_FILE)
    settings, analysis_paths = prepare_analysis(settings_file)
    WORKER_SETTINGS_FILE = settings_file
    ACTIVE_BOUNDS = get_active_bounds(settings)
    RAW_KEYS = get_raw_variables(settings)
    LABELS_HEADER = ", ".join(RAW_KEYS)
    SAVE_DIR = Path(analysis_paths["data"]) / DATA_TAG
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = SAVE_DIR / OUTPUT_FILE
    return settings, analysis_paths


def initialize_output_file():
    if OUTPUT_PATH is None:
        raise RuntimeError("OUTPUT_PATH is not configured. Call configure_analysis() first.")

    if OUTPUT_PATH.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(
                f"Output file already exists: {OUTPUT_PATH}. "
                "Change OUTPUT_FILE or set OVERWRITE_OUTPUT=True."
            )
        OUTPUT_PATH.unlink()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(f"# {LABELS_HEADER}\n", encoding="utf-8")


def append_row(row):
    row = np.asarray(row, dtype=float).reshape(1, -1)
    validate_row_width(RAW_KEYS, row[0], "scripts/run_global_scan.py")
    with OUTPUT_PATH.open("ab") as handle:
        np.savetxt(handle, row, fmt="%.6e")


def evaluate_point(params, raw_keys=None):
    selected_keys = raw_keys or RAW_KEYS
    evaluated = evaluate_raw(params, raw_keys=selected_keys)
    if evaluated is None:
        return None

    _, cost_function, _, row = evaluated

    if cost_function > PENALTY_MAX:
        return None

    row = np.asarray(row, dtype=float)
    validate_row_width(selected_keys, row, "scripts/run_global_scan.py")
    return row


def evaluate_point_task(task):
    params, raw_keys = task
    return evaluate_point(params, raw_keys=raw_keys)


def get_lhs_samples(n_samples, param_bounds):
    dim = len(param_bounds)
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=n_samples)
    low = np.array([b[0] for b in param_bounds])
    high = np.array([b[1] for b in param_bounds])
    return qmc.scale(sample, low, high)


def main():
    configure_analysis()

    print(f"\n=== RANDOM RAW SCANNER ===")
    print(f"Target: {TARGET_POINTS} valid points.")
    print(f"Fail-safe limit: {MAX_EVALUATIONS} evaluations.")
    print(f"Output: {OUTPUT_PATH}")

    initialize_output_file()

    t0 = time.time()

    saved_points = 0
    cores = max(1, cpu_count() - 1)
    total_evaluated = 0

    # Bucle continuo hasta alcanzar la cuota o el limite de seguridad
    while saved_points < TARGET_POINTS and total_evaluated < MAX_EVALUATIONS:

        # Ajustamos el tamano del chunk si estamos cerca del limite de seguridad
        evals_left = MAX_EVALUATIONS - total_evaluated
        current_chunk = min(CHUNK_SIZE, evals_left)

        seeds = get_lhs_samples(current_chunk, ACTIVE_BOUNDS)
        total_evaluated += current_chunk

        with Pool(
            processes=cores,
            initializer=init_worker,
            initargs=(WORKER_SETTINGS_FILE,),
        ) as pool:
            tasks = ((seed, RAW_KEYS) for seed in seeds)
            results = pool.imap_unordered(evaluate_point_task, tasks, chunksize=500)

            for res in results:
                if res is not None:
                    append_row(res)
                    saved_points += 1
                    # Detenemos inmediatamente los procesos paralelos si alcanzamos la meta
                    if saved_points >= TARGET_POINTS:
                        pool.terminate()
                        break

        print(f"  -> Evaluated: {total_evaluated}/{MAX_EVALUATIONS} | Valid: {saved_points}/{TARGET_POINTS}")

    if saved_points == 0:
        print("\n[!] Cero puntos encontrados. Revisa PENALTY_MAX, bounds o incrementa MAX_EVALUATIONS.")
        return

    dt = time.time() - t0
    print(f"\n=== SCAN DONE ===")
    print(f"File saved: {OUTPUT_PATH.name}")
    print(f"Points exported: {saved_points}")
    print(f"Total time: {int(dt//60)}m {int(dt%60)}s")
    print(f"Acceptance rate: {(saved_points / total_evaluated)*100:.4f}%")


if __name__ == "__main__":
    main()
