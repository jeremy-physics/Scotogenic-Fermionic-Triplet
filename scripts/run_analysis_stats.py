import os
import sys
import numpy as np
from pathlib import Path


# === USER/AGENT CONFIGURATION START ===

USE_CURRENT_ANALYSIS = True
LOCAL_SETTINGS_FILE = None

DATA_TAG = "scans"
INPUT_FILE = "sampler_01.dat"

OUTPUT_TAG = "stats"
OUTPUT_FILE = "stats_sampler_01.txt"

# Correlation Threshold
THRESHOLD = 0.7

# Optional simple filter. Leave FILTER_MAX/FILTER_MIN as None to disable.
FILTER_COLUMN = "cost_function"
FILTER_MIN = None
FILTER_MAX = None

# Statistical target selection.
STAT_TARGET_COLUMN = "chi"
FALLBACK_STAT_TARGET_COLUMN = "cost_function"
TARGET_FOR_CORRELATIONS = STAT_TARGET_COLUMN
ZERO_TOL = 1e-12

# === USER/AGENT CONFIGURATION END ===

# --- Path Configuration ---
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = CURRENT_DIR.parent
PROJECT_SRC = PROJECT_ROOT / "src"

module_path = str(PROJECT_SRC)
if module_path not in sys.path:
    sys.path.append(module_path)

from io_tools import get_settings_file, prepare_analysis
from variables import get_raw_variables


INPUT_PATH = None
OUTPUT_PATH = None
CONFIGURED_RAW_LABELS = None


def configure_analysis():
    global CONFIGURED_RAW_LABELS, INPUT_PATH, OUTPUT_PATH

    settings_file = get_settings_file(USE_CURRENT_ANALYSIS, LOCAL_SETTINGS_FILE)
    settings, analysis_paths = prepare_analysis(settings_file)
    CONFIGURED_RAW_LABELS = get_raw_variables(settings)
    input_dir = Path(analysis_paths["data"]) / DATA_TAG
    output_dir = Path(analysis_paths["data"]) / OUTPUT_TAG
    output_dir.mkdir(parents=True, exist_ok=True)

    INPUT_PATH = input_dir / INPUT_FILE
    OUTPUT_PATH = output_dir / OUTPUT_FILE
    return settings, analysis_paths


def parse_header(data_file):
    headers = []
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("#"):
                    break
                header = stripped.lstrip("#").strip()
                if header:
                    headers.append(header)
    except OSError as e:
        print(f"[WARN] Could not read header from {data_file}: {e}")
        return None

    if not headers:
        return None

    header = headers[-1]
    if "," in header:
        return [item.strip() for item in header.split(",") if item.strip()]
    return header.split()


def load_data(data_file):
    if not data_file.exists():
        print(f"[ERROR] Data not found: {data_file}")
        return None, None

    print(f"Reading: {data_file.name}...")
    try:
        data = np.loadtxt(data_file, dtype=float, comments="#")
    except Exception as e:
        print(f"[ERROR] Read failed: {e}")
        return None, None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.size == 0:
        print(f"[ERROR] Data file is empty: {data_file}")
        return None, None

    n_cols = data.shape[1]
    header_labels = parse_header(data_file)

    if header_labels is None:
        print("[WARN] No header found. Using fallback/generic labels.")
        labels = CONFIGURED_RAW_LABELS if len(CONFIGURED_RAW_LABELS or []) == n_cols else [f"Col_{i}" for i in range(n_cols)]
    elif len(header_labels) != n_cols:
        print(f"[WARN] Header column mismatch ({len(header_labels)} labels vs {n_cols} columns). Using fallback/generic labels.")
        labels = CONFIGURED_RAW_LABELS if len(CONFIGURED_RAW_LABELS or []) == n_cols else [f"Col_{i}" for i in range(n_cols)]
    else:
        labels = header_labels

    return data, labels


def apply_filter(data, labels):
    if FILTER_COLUMN is None or (FILTER_MIN is None and FILTER_MAX is None):
        return data

    if FILTER_COLUMN not in labels:
        print(f"[WARN] Filter column not found: {FILTER_COLUMN}. Filter skipped.")
        return data

    idx = labels.index(FILTER_COLUMN)
    mask = np.ones(data.shape[0], dtype=bool)

    if FILTER_MIN is not None:
        mask &= data[:, idx] >= FILTER_MIN
    if FILTER_MAX is not None:
        mask &= data[:, idx] < FILTER_MAX

    filtered = data[mask]
    print(f"Filter: {len(filtered)}/{len(data)} rows kept.")
    return filtered


def resolve_stat_target(labels):
    messages = []

    if STAT_TARGET_COLUMN in labels:
        stat_target = STAT_TARGET_COLUMN
    elif FALLBACK_STAT_TARGET_COLUMN in labels:
        stat_target = FALLBACK_STAT_TARGET_COLUMN
        messages.append(
            f"[WARN] {STAT_TARGET_COLUMN} no existe en este archivo; "
            f"usando {FALLBACK_STAT_TARGET_COLUMN} como fallback."
        )
    else:
        stat_target = None
        messages.append(
            f"[WARN] No se encontro {STAT_TARGET_COLUMN} ni "
            f"{FALLBACK_STAT_TARGET_COLUMN}; no hay columna estadistica principal."
        )

    requested_corr_target = TARGET_FOR_CORRELATIONS or stat_target
    if requested_corr_target in labels:
        corr_target = requested_corr_target
    elif stat_target in labels:
        corr_target = stat_target
        if not (
            requested_corr_target == STAT_TARGET_COLUMN
            and stat_target == FALLBACK_STAT_TARGET_COLUMN
            and STAT_TARGET_COLUMN not in labels
        ):
            messages.append(
                f"[WARN] Target for correlations not found: {requested_corr_target}. "
                f"Using {stat_target}."
            )
    else:
        corr_target = None

    return stat_target, corr_target, messages


def basic_stats_df(data, labels):
    stats = {
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0, ddof=1) if data.shape[0] > 1 else np.zeros(data.shape[1]),
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
    }

    name_width = max(len(label) for label in labels)
    lines = [f"{'column':<{name_width}}  {'mean':>14}  {'std':>14}  {'min':>14}  {'max':>14}"]
    for i, label in enumerate(labels):
        lines.append(
            f"{label:<{name_width}}  "
            f"{stats['mean'][i]:>14.6e}  "
            f"{stats['std'][i]:>14.6e}  "
            f"{stats['min'][i]:>14.6e}  "
            f"{stats['max'][i]:>14.6e}"
        )
    return "\n".join(lines)


def scalar_stats_line(name, values):
    return (
        f"{name:<14} "
        f"min={np.min(values):.6e}  "
        f"max={np.max(values):.6e}  "
        f"median={np.median(values):.6e}"
    )


def cost_breakdown_lines(data, labels, stat_target, messages):
    lines = ["\n--- Cost Columns ---"]

    for message in messages:
        lines.append(message)

    if stat_target is not None:
        lines.append(f"Stat target: {stat_target}")

    recognized = [
        name for name in ("chi", "pen_structure", "pen_restriction", "penalty_total", "cost_function")
        if name in labels
    ]
    if recognized:
        lines.append("Recognized: " + ", ".join(recognized))
    else:
        lines.append("[WARN] No chi, penalty or cost_function column found.")
        return lines

    for name in ("chi", "pen_structure", "pen_restriction", "penalty_total", "cost_function"):
        if name not in labels:
            continue
        values = data[:, labels.index(name)]
        lines.append(scalar_stats_line(name, values))

        if name == "penalty_total":
            near_zero = np.isclose(values, 0.0, atol=ZERO_TOL, rtol=0.0)
            positive = values > ZERO_TOL
            lines.append(f"{name:<14} near_zero={int(np.sum(near_zero))}  positive={int(np.sum(positive))}")

    return lines


def rankdata_average(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end

    return ranks


def spearman_corrcoef(data):
    ranked = np.apply_along_axis(rankdata_average, 0, data)
    return np.corrcoef(ranked, rowvar=False)


def correlation_lines(corr, labels, title, value_label):
    lines = [f"\n--- {title} (|{value_label}| > {THRESHOLD}) ---"]
    found = False

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            val = corr[i, j]
            if np.isfinite(val) and abs(val) > THRESHOLD:
                lines.append(f"{labels[i]:<12} vs {labels[j]:<12}: {value_label} = {val:.3f}")
                found = True

    if not found:
        lines.append("None found.")

    return lines


def target_correlation_lines(corr, labels, target_label, title, value_label):
    lines = [f"\n--- {title}: target = {target_label} (|{value_label}| > {THRESHOLD}) ---"]

    if target_label is None or target_label not in labels:
        lines.append("[WARN] No valid target column for correlations.")
        return lines

    target_idx = labels.index(target_label)
    items = []
    for idx, label in enumerate(labels):
        if idx == target_idx:
            continue
        val = corr[target_idx, idx]
        if np.isfinite(val) and abs(val) > THRESHOLD:
            items.append((abs(val), label, val))

    if not items:
        lines.append("None found.")
        return lines

    for _, label, val in sorted(items, reverse=True):
        lines.append(f"{target_label:<12} vs {label:<12}: {value_label} = {val:.3f}")

    return lines


def build_report(data, labels, input_path):
    stat_target, corr_target, target_messages = resolve_stat_target(labels)

    lines = [
        "=== ANALYSIS STATS ===",
        f"Input file: {input_path}",
        f"Rows: {data.shape[0]}",
        f"Columns: {data.shape[1]}",
        "",
        "--- Columns ---",
        ", ".join(labels),
        "",
        "--- Basic Statistics ---",
        basic_stats_df(data, labels),
    ]
    lines.extend(cost_breakdown_lines(data, labels, stat_target, target_messages))

    if data.shape[0] < 2:
        lines.append("\n[WARN] At least two rows are required for correlations.")
        return lines

    pear = np.corrcoef(data, rowvar=False)
    rho = spearman_corrcoef(data)

    lines.extend(target_correlation_lines(pear, labels, corr_target, "Pearson Correlations", "r"))
    lines.extend(target_correlation_lines(rho, labels, corr_target, "Spearman Correlations", "rho"))
    lines.extend(correlation_lines(pear, labels, "Pearson Correlations", "r"))
    lines.extend(correlation_lines(rho, labels, "Spearman Correlations", "rho"))

    return lines


def main():
    configure_analysis()
    print("\n=== CORRELATION ANALYZER ===")

    data, labels = load_data(INPUT_PATH)
    if data is None:
        return

    data = apply_filter(data, labels)
    if data.size == 0:
        print("[ERROR] No rows left after filtering.")
        return

    print("Calculating statistics...")
    report = build_report(data, labels, INPUT_PATH)
    text = "\n".join(report) + "\n"

    print(text)
    try:
        OUTPUT_PATH.write_text(text, encoding="utf-8")
        print(f"[SAVE] Stats saved to: {OUTPUT_PATH}")
    except OSError as e:
        print(f"[ERROR] Could not write stats file: {e}")


if __name__ == "__main__":
    main()
