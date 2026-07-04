import os
import sys
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from scipy.optimize import differential_evolution
from multiprocessing import freeze_support


# === USER/AGENT CONFIGURATION START ===

USE_CURRENT_ANALYSIS = True
LOCAL_SETTINGS_FILE = None

# Choose the seed/refined files manually here.
ROOT = "01"
SEED_FILE = f"point_{ROOT}.dat"
OUTPUT_FILE = f"point_{ROOT}_R.dat"

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
N_POP = 10
THRESHOLD = 1e0
EXPLORE_SEEDS = [101, 202, 303, 404, 505]

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


def coerce_seed(seed_vec, bounds):
    seed_vec = np.asarray(seed_vec, float).ravel()
    dim = len(bounds)
    low, high = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    if seed_vec.size < dim:
        full = 0.5 * (low + high)
        if seed_vec.size > 0: full[:seed_vec.size] = seed_vec
        return full
    return np.clip(seed_vec[:dim], low, high)


# --- Population Generators ---
def make_init_phaseA(seed_vec, bounds, popsize_per_dim=18, frac_local=0.6, jitter=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    dim = len(bounds)
    pop = max(dim * popsize_per_dim, 5)
    n_loc = int(frac_local * pop)

    low, high = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    # Local cloud + Global random
    loc = seed_vec + rng.uniform(-jitter*(high-low), jitter*(high-low), size=(n_loc, dim))
    rnd = rng.uniform(low, high, size=(pop - n_loc, dim))

    return np.vstack([np.clip(loc, low, high), rnd])


def narrow_pop(xbest, bounds, width=0.05, pop_per_dim=15, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    dim = len(bounds)
    pop = max(dim * pop_per_dim, 5)
    low, high = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    cloud = xbest + rng.uniform(-width*(high-low), width*(high-low), size=(pop-1, dim))
    return np.vstack([xbest[None, :], np.clip(cloud, low, high)])


# --- Optimization ---
def run_phase_A(seed_vec, bounds, seed, maxiter=200, popsize=20):
    """ Exploration Phase """
    initA = make_init_phaseA(seed_vec, bounds, popsize_per_dim=popsize)
    best = {"cost_function": np.inf, "x": seed_vec}

    def callback(xk, convergence):
        cost_function = chisquare_wrapper(xk)
        if cost_function < best["cost_function"]:
            best["cost_function"] = cost_function
            best["x"] = xk
        return cost_function < THRESHOLD

    res = differential_evolution(
        chisquare_wrapper, bounds, strategy='rand2bin', mutation=(0.6, 1.6), recombination=0.9,
        popsize=popsize, maxiter=maxiter, init=initA, workers=-1, updating='deferred',
        polish=False, callback=callback, seed=seed, tol=1e-4
    )
    return (res.x, res.fun) if res.success else (best["x"], best["cost_function"])


def run_refinement(seed_vector, output_name, maxiter=10**10):
    """ Refinement Phase """
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
    if np.isfinite(res.fun) and res.fun < best.get("cost_function", np.inf):
        best["cost_function"] = float(res.fun)
        best["x"] = np.asarray(res.x, dtype=float)
        save_to_file(output_name, best["x"], best["cost_function"])

    dt = time.time() - t0
    print(f"--> Refinement done in {int(dt//60)}m {int(dt%60)}s.")
    return res


# --- Main ---
def main():
    configure_analysis()

    print(f"\n=== SINGLE OPTIMIZATION ===\nDir: {SAVE_DIR}\n")

    # Load Seed
    seed_vec = coerce_seed(np.array([]), ACTIVE_BOUNDS)
    for f in [OUTPUT_FILE, SEED_FILE]:
        p = SAVE_DIR / f

        if p.exists():
            try:
                seed_vec = coerce_seed(np.loadtxt(p, comments="#"), ACTIVE_BOUNDS)
                print(f"Loaded seed: {p.name}")
                break
            except: pass

    # Phase A (Exploration) - Optional
    # Uncomment block below to enable island exploration
    """
    candidates = []
    print("Running Phase A (Exploration)...")
    for sd in EXPLORE_SEEDS:
        print(f"  -> Island {sd}...", end="\r")
        x, c2 = run_phase_A(seed_vec, ACTIVE_BOUNDS, seed=sd)
        print(f"  -> Island {sd}: cost_function = {c2:.6e}")
        candidates.append((c2, x, sd))
        if c2 < THRESHOLD: break

    candidates.sort(key=lambda x: x[0])
    best_cost, seed_vec, best_seed = candidates[0]
    print(f"\nBest Island: {best_seed} (cost_function={best_cost:.6e})")
    """

    # Phase B (Refinement)
    print(f"Refining target: {OUTPUT_FILE}...")
    res = run_refinement(seed_vec, OUTPUT_FILE)
    print(f"\n=== DONE ===\nFinal cost_function: {res.fun:.6e}")


if __name__ == "__main__":
    freeze_support()
    try: main()
    except KeyboardInterrupt: print("\nAborted.")
