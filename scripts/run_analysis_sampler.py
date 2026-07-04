import os
import sys
import numpy as np
from pathlib import Path


# === USER/AGENT CONFIGURATION START ===

USE_CURRENT_ANALYSIS = True
LOCAL_SETTINGS_FILE = None
SEED_FILE = "point_01.dat"

DATA_TAG = "scans"
OUTPUT_FILE = "sampler_01.dat"

# --- Sampler Settings ---
KAPPA = 0.1
MAX_POINTS = 1000
MAX_ATTEMPTS_FACTOR = 100

RAW_HEADER = None

# === USER/AGENT CONFIGURATION END ===

# --- Path Configuration ---
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = CURRENT_DIR.parent
PROJECT_SRC = PROJECT_ROOT / "src"

module_path = str(PROJECT_SRC)
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from io_tools import get_settings_file, prepare_analysis
from likelihood import bounds, evaluate_raw, LOG_IDX, MAX_EXP, MIN_EXP, to_physical
from bounds import get_active_bounds
from variables import get_raw_variables, validate_row_width


OPTIMIZATION_DIR = None
SAVE_DIR = None
OUTPUT_PATH = None
RAW_KEYS = None
ACTIVE_BOUNDS = bounds


def configure_analysis():
    global ACTIVE_BOUNDS, OPTIMIZATION_DIR, RAW_HEADER, RAW_KEYS, SAVE_DIR, OUTPUT_PATH

    settings_file = get_settings_file(USE_CURRENT_ANALYSIS, LOCAL_SETTINGS_FILE)
    settings, analysis_paths = prepare_analysis(settings_file)
    ACTIVE_BOUNDS = get_active_bounds(settings)
    RAW_KEYS = get_raw_variables(settings)
    RAW_HEADER = ", ".join(RAW_KEYS)
    OPTIMIZATION_DIR = Path(analysis_paths["optimization"])
    SAVE_DIR = Path(analysis_paths["data"]) / DATA_TAG
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = SAVE_DIR / OUTPUT_FILE
    return settings, analysis_paths


def load_seed_scan(seed_path):
    raw = np.loadtxt(seed_path, comments="#")
    seed = raw[-1] if raw.ndim > 1 else raw
    return np.asarray(seed, dtype=float)


def physical_to_scan(point_physical):
    point_physical = np.asarray(point_physical, dtype=float).copy()
    point_scan = point_physical.copy()

    if point_scan.size != len(ACTIVE_BOUNDS):
        print(f"[WARN] Perturbed point has {point_scan.size} parameters; expected {len(ACTIVE_BOUNDS)}.")
        return None

    low = np.array([b[0] for b in ACTIVE_BOUNDS], dtype=float)
    high = np.array([b[1] for b in ACTIVE_BOUNDS], dtype=float)

    for map_pos, idx in enumerate(LOG_IDX):
        value = point_physical[idx]
        if not np.isfinite(value) or value == 0.0:
            return None

        exponent = np.log10(abs(value))
        span = MAX_EXP[map_pos] - MIN_EXP[map_pos]
        if span == 0:
            return None

        magnitude = (exponent - MIN_EXP[map_pos]) / span
        point_scan[idx] = np.sign(value) * magnitude

    if not np.all(np.isfinite(point_scan)):
        return None

    eps = 1e-12
    if np.any(point_scan < low - eps) or np.any(point_scan > high + eps):
        return None

    return np.clip(point_scan, low, high)


def perturb_physical(point_physical):
    factors = np.random.uniform(1 - KAPPA, 1 + KAPPA, size=point_physical.size)
    return point_physical * factors


def main():
    configure_analysis()

    seed_path = OPTIMIZATION_DIR / SEED_FILE
    if not seed_path.exists():
        print(f"[ERROR] Seed file not found: {seed_path}")
        return

    print(f"\n=== ANALYSIS SAMPLER ===")
    print(f"Loading seed: {seed_path.name}")

    try:
        seed_scan = load_seed_scan(seed_path)
    except Exception as e:
        print(f"[ERROR] Failed to load seed: {e}")
        return

    if seed_scan.size != len(ACTIVE_BOUNDS):
        print(f"[ERROR] Seed has {seed_scan.size} parameters; expected {len(ACTIVE_BOUNDS)}.")
        return

    seed_physical = to_physical(seed_scan)
    print(f"Generating {MAX_POINTS} points (Physical perturbation: {KAPPA*100}%)...")

    raw_output = []
    rejected = 0
    count = 0
    attempts = 0
    max_attempts = MAX_ATTEMPTS_FACTOR * MAX_POINTS

    while len(raw_output) < MAX_POINTS and attempts < max_attempts:
        attempts += 1
        point_physical = perturb_physical(seed_physical)
        point_scan = physical_to_scan(point_physical)

        if point_scan is None:
            rejected += 1
            continue

        try:
            evaluated = evaluate_raw(point_scan, raw_keys=RAW_KEYS)
            if evaluated is None:
                rejected += 1
                continue

            _, _, _, row = evaluated
            raw_row = np.asarray(row, dtype=float)
            validate_row_width(RAW_KEYS, raw_row, "scripts/run_analysis_sampler.py")
            if not np.all(np.isfinite(raw_row)):
                print("nan encontrado")
                rejected += 1
                continue

            raw_output.append(raw_row.ravel())

            count += 1
            if count % 500 == 0:
                print(f"  -> {count}/{MAX_POINTS}...", end="\r")

        except Exception:
            rejected += 1
            continue

    if len(raw_output) < MAX_POINTS and attempts >= max_attempts:
        print("Sampler stopped after MAX_ATTEMPTS.")
        print(f"Accepted points: {len(raw_output)}")
        print(f"Rejected points: {rejected}")

    if not raw_output:
        print("No valid sampled points were produced.")
        return

    raw_matrix = np.asarray(raw_output)
    validate_row_width(RAW_KEYS, raw_matrix[0], "scripts/run_analysis_sampler.py")

    np.savetxt(OUTPUT_PATH, raw_matrix, fmt="%.6e", header=RAW_HEADER)

    print(f"\n[DONE] Saved {len(raw_output)} raw rows to {OUTPUT_PATH.name}")
    print(f"[INFO] Rejected perturbations: {rejected}")


if __name__ == "__main__":
    main()
