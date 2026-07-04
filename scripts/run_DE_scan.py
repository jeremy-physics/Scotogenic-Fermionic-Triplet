import os
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from scipy.stats import qmc
from scipy.optimize import differential_evolution


# === USER/AGENT CONFIGURATION START ===

USE_CURRENT_ANALYSIS = True
LOCAL_SETTINGS_FILE = None
DATA_TAG = "scans"
OUTPUT_FILE = "scanDE_01.dat"

# --- USER SETTINGS ---
CHUNK_SIZE = 10
TARGET_POINTS = 1
MAX_EVALUATIONS = 10000000
LABELS_HEADER = None

TH_TRIGGER_DE = 1000
TH_ACCEPT = 1000
TH_PENALTY_WALL = 1e20

# === DE SETTINGS ===
N_POP = 8
MAX_ITERS = 200

# === USER/AGENT CONFIGURATION END ===

# --- Path Configuration ---
CURRENT_DIR = Path(os.path.abspath(__file__)).parent
PROJECT_ROOT = CURRENT_DIR.parent
PROJECT_SRC = PROJECT_ROOT / "src"

module_path = str(PROJECT_SRC)
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from io_tools import get_settings_file, prepare_analysis
from likelihood import bounds, chisquare_wrapper, evaluate_raw
from bounds import get_active_bounds
from variables import get_raw_variables, validate_row_width


# === CONFIGURATION ===
SAVE_DIR = None
OUTPUT_PATH = None
RAW_KEYS = None
ACTIVE_BOUNDS = bounds


def configure_analysis():
    global ACTIVE_BOUNDS, LABELS_HEADER, RAW_KEYS, SAVE_DIR, OUTPUT_PATH

    settings_file = get_settings_file(USE_CURRENT_ANALYSIS, LOCAL_SETTINGS_FILE)
    settings, analysis_paths = prepare_analysis(settings_file)
    ACTIVE_BOUNDS = get_active_bounds(settings)
    RAW_KEYS = get_raw_variables(settings)
    LABELS_HEADER = ", ".join(RAW_KEYS)
    SAVE_DIR = Path(analysis_paths["data"]) / DATA_TAG
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = SAVE_DIR / OUTPUT_FILE
    return settings, analysis_paths


def validate_existing_output_header(output_path):
    try:
        with open(output_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("#"):
                    raise ValueError(
                        f"[ERROR] scripts/run_DE_scan.py: existing append file has no header: "
                        f"{output_path}"
                    )
                header = stripped.lstrip("#").strip()
                labels = [item.strip() for item in header.split(",") if item.strip()]
                if labels != RAW_KEYS:
                    raise ValueError(
                        f"[ERROR] scripts/run_DE_scan.py: existing header has "
                        f"{len(labels)} columns but the configured header has "
                        f"{len(RAW_KEYS)}. Requested keys: {RAW_KEYS}"
                    )
                return
    except OSError as exc:
        raise ValueError(
            f"[ERROR] scripts/run_DE_scan.py: could not validate existing output "
            f"{output_path}: {exc}"
        ) from exc

    raise ValueError(
        f"[ERROR] scripts/run_DE_scan.py: existing append file is empty: {output_path}"
    )


def narrow_pop(xbest, bounds, width=0.05, pop_per_dim=5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    dim = len(bounds)
    pop = max(dim * pop_per_dim, 5)
    low, high = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    cloud = xbest + rng.uniform(-width*(high-low), width*(high-low), size=(pop-1, dim))
    return np.vstack([xbest[None, :], np.clip(cloud, low, high)])


def evaluate_and_optimize(seed):
    cost_init = chisquare_wrapper(seed)

    if cost_init >= TH_PENALTY_WALL:
        return None

    best_x, best_cost = seed, cost_init

    if cost_init > TH_TRIGGER_DE:
        init_pop = narrow_pop(seed, ACTIVE_BOUNDS)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = differential_evolution(
                chisquare_wrapper, ACTIVE_BOUNDS, strategy='best1bin', mutation=(0.4, 1.0), recombination=0.9,
                init=init_pop, maxiter=MAX_ITERS, popsize=N_POP,
                workers=-1, updating='deferred', tol=1e-8
            )
            best_x, best_cost = res.x, res.fun

    if best_cost <= TH_ACCEPT:
        evaluated = evaluate_raw(best_x, raw_keys=RAW_KEYS)
        if evaluated is not None:
            row = np.asarray(evaluated[3], dtype=float)
            validate_row_width(RAW_KEYS, row, "scripts/run_DE_scan.py")
            return row

    return None


def get_lhs_samples(n_samples, param_bounds):
    dim = len(param_bounds)
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=n_samples)
    low, high = np.array([b[0] for b in param_bounds]), np.array([b[1] for b in param_bounds])
    return qmc.scale(sample, low, high)


def main():
    configure_analysis()

    print(f"\n=== SMART RAW SCANNER (SEQUENTIAL) ===")
    print(f"Target: {TARGET_POINTS} points (cost_function <= {TH_ACCEPT})")
    print(f"Output: {OUTPUT_PATH}")

    t0 = time.time()
    total_evaluated = 0
    total_saved = 0

    # Check si el archivo ya existe para no duplicar el header si el script se interrumpio y reinicio
    file_exists = OUTPUT_PATH.exists()
    if file_exists:
        validate_existing_output_header(OUTPUT_PATH)

    while total_saved < TARGET_POINTS and total_evaluated < MAX_EVALUATIONS:
        evals_left = MAX_EVALUATIONS - total_evaluated
        current_chunk = min(CHUNK_SIZE, evals_left)

        seeds = get_lhs_samples(current_chunk, ACTIVE_BOUNDS)
        chunk_valid_data = []

        for seed in seeds:
            total_evaluated += 1
            res = evaluate_and_optimize(seed)

            if res is not None:
                chunk_valid_data.append(res)
                total_saved += 1
                print(f"  [+] Valid Point Found! ({total_saved}/{TARGET_POINTS}) - Eval #{total_evaluated}")

                if total_saved >= TARGET_POINTS:
                    break

        # --- VOLCADO A DISCO POR CHUNK ---
        if chunk_valid_data:
            chunk_matrix = np.vstack(chunk_valid_data)
            validate_row_width(RAW_KEYS, chunk_matrix[0], "scripts/run_DE_scan.py")
            with open(OUTPUT_PATH, "a") as f:
                if not file_exists:
                    f.write(f"# {LABELS_HEADER}\n")
                    file_exists = True

                # Volcado matricial del lote
                np.savetxt(f, chunk_matrix, fmt="%.6e")

        print(f"  -> LHS Seeds evaluated: {total_evaluated}/{MAX_EVALUATIONS}")

        if total_evaluated >= MAX_EVALUATIONS:
            break

    dt = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"Exported {total_saved} points to: {OUTPUT_PATH.name}")
    print(f"Total processing time: {int(dt//60)}m {int(dt%60)}s")


if __name__ == "__main__":
    main()
