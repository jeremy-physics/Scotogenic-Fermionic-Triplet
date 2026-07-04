import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from multiprocessing import freeze_support


# === USER/AGENT CONFIGURATION START ===

USE_CURRENT_ANALYSIS = True
LOCAL_SETTINGS_FILE = None

# === USER/AGENT CONFIGURATION END ===

# --- Path Configuration ---
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = CURRENT_DIR.parent
PROJECT_SRC = PROJECT_ROOT / "src"

module_path = str(PROJECT_SRC)
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from io_tools import get_settings_file, prepare_analysis

# --- Physics Imports ---
from constants import *
from likelihood import *
from bounds import get_active_bounds

# --- Configuration ---
SAVE_DIR = None
ACTIVE_BOUNDS = bounds

# === USER/AGENT CONFIGURATION START ===

# Settings
N_POP = 8
THRESHOLD = 1.0
NUM_SEEDS = 1
BENCHMARK_PREFIX = "point_"
RANDOM_SEED = 20260703
EXPLICIT_SEED_POINTS = [
    [
        7.36766567230249469489e-01,
        2.19856478608236649563e-01,
        -5.44882386962671994013e-01,
        6.65780316404862348989e-01,
        -7.87569512137278326946e-01,
        -8.00570686219472027467e-01,
        9.40640188541980393211e-01,
        8.11892045492145886243e-01,
        9.18934454144865631164e-01,
        8.64429764539270673041e-01,
        -1.88587217292998987617e-01,
        7.15979915853515169744e-01,
        2.15312356011795991151e-01,
        2.99397855611312824209e-01,
        -1.72319863762671499074e-01,
        5.51992856172134005455e-01,
        3.14870833520645510006e+00,
    ],
]

# === USER/AGENT CONFIGURATION END ===


def configure_analysis():
    global ACTIVE_BOUNDS, SAVE_DIR

    settings_file = get_settings_file(USE_CURRENT_ANALYSIS, LOCAL_SETTINGS_FILE)
    settings, analysis_paths = prepare_analysis(settings_file)
    ACTIVE_BOUNDS = get_active_bounds(settings)
    SAVE_DIR = Path(analysis_paths["optimization"])
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    return settings, analysis_paths


# --- Helpers ---
def save_to_file(filename, x, cost_function):
    if SAVE_DIR is None:
        raise RuntimeError("SAVE_DIR is not configured. Call configure_analysis() first.")

    try:
        path = SAVE_DIR / filename
        content = f"# timestamp: {datetime.now()}\n# cost_function: {cost_function:.20e}\n" + \
                  " ".join(f"{v:.20e}" for v in np.asarray(x, float)) + "\n"
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        print(f"[ERROR] Save failed: {e}")


# --- Population Generators ---
def get_lhs_seeds(n_seeds, bounds, random_seed=None):
    """ Latin Hypercube Sampling """
    dim = len(bounds)
    sampler = qmc.LatinHypercube(d=dim, seed=random_seed)
    sample = sampler.random(n=n_seeds)
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    return qmc.scale(sample, low, high)


def get_configured_seeds():
    if EXPLICIT_SEED_POINTS:
        seeds = np.asarray(EXPLICIT_SEED_POINTS, dtype=float)
        if seeds.ndim == 1:
            seeds = seeds[None, :]
        if seeds.shape[1] != len(ACTIVE_BOUNDS):
            raise ValueError(
                f"EXPLICIT_SEED_POINTS has {seeds.shape[1]} parameters; "
                f"expected {len(ACTIVE_BOUNDS)}."
            )
        return seeds
    return get_lhs_seeds(NUM_SEEDS, ACTIVE_BOUNDS, RANDOM_SEED)


def narrow_pop(xbest, bounds, width=0.05, pop_per_dim=15, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    dim = len(bounds)
    pop = max(dim * pop_per_dim, 5)
    low, high = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    cloud = xbest + rng.uniform(-width*(high-low), width*(high-low), size=(pop-1, dim))
    return np.vstack([xbest[None, :], np.clip(cloud, low, high)])


# --- Optimization ---
def run_refinement(seed_vector, output_name, maxiter=2500):
    init_pop = narrow_pop(seed_vector, ACTIVE_BOUNDS)
    best = {"cost_function": np.inf}

    def callback(xk, convergence):
        cost_function = chisquare_wrapper(xk)
        if cost_function < best["cost_function"]:
            best["cost_function"] = cost_function
            save_to_file(output_name, xk, cost_function)
        return cost_function < THRESHOLD

    t0 = time.time()
    res = differential_evolution(
        chisquare_wrapper, ACTIVE_BOUNDS, strategy='best1bin', mutation=(0.4, 1.0), recombination=0.9,
        init=init_pop, maxiter=maxiter, popsize=N_POP, polish=True, workers=-1, updating='deferred',
        callback=callback, tol=1e-8
    )
    dt = time.time() - t0
    print(f"--> Refinement done in {int(dt//60)}m {int(dt%60)}s.")

    # Ensure final best is saved
    if res.fun < best["cost_function"]:
        save_to_file(output_name, res.x, res.fun)
    return res


# --- Main ---
def main():
    configure_analysis()

    print(f"\n=== SWEEP START ===\nDir: {SAVE_DIR}")
    print(f"Generating {NUM_SEEDS} configured seed(s)...\n")

    t_total = time.time()
    seeds = get_configured_seeds()

    for i, seed in enumerate(seeds):
        filename = f"{BENCHMARK_PREFIX}{i+1:02d}.dat"
        print(f">> Optimizing {filename}...")
        run_refinement(seed, output_name=filename)

    dt = time.time() - t_total
    print(f"\n=== SWEEP DONE ===\nTotal time: {int(dt//60)}m {int(dt%60)}s.")


if __name__ == "__main__":
    freeze_support()
    try: main()
    except KeyboardInterrupt: print("\nAborted.")
