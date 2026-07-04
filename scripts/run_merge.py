import os
import sys
import numpy as np
from pathlib import Path


# === USER/AGENT CONFIGURATION START ===

USE_CURRENT_ANALYSIS = True
LOCAL_SETTINGS_FILE = None

MERGE_MODE = "benchmark_points"  # "data_files" or "benchmark_points"

# data_files mode: discover files below the active Analysis.
INPUT_FOLDER = "data/scans"
FILE_PATTERN = "*.dat"

# Optional compatibility override. When non-empty, this manual list is used
# instead of INPUT_FOLDER/FILE_PATTERN. Entries keep the form (DATA_TAG, file).
INPUT_FILES = []

POINT_PREFIX = "point"
EXCLUDE_REFINED = False
INCLUDE_ONLY_REFINED = False
BENCHMARK_FILES = []

OUTPUT_FOLDER = "data/merged"
OUTPUT_FILE = "merged_points_01.dat"

STRICT_HEADER_CHECK = True
ALLOW_COMMON_COLUMNS_MERGE = False

# Used only when no compatible header is available.
RAW_HEADER = None

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


OUTPUT_PATH = None
ANALYSIS_DIR = None
DATA_ROOT = None
OPTIMIZATION_DIR = None


def configure_analysis():
    global ANALYSIS_DIR, DATA_ROOT, OPTIMIZATION_DIR, OUTPUT_PATH, RAW_HEADER

    settings_file = get_settings_file(USE_CURRENT_ANALYSIS, LOCAL_SETTINGS_FILE)
    settings, analysis_paths = prepare_analysis(settings_file)
    RAW_HEADER = ", ".join(get_raw_variables(settings))
    ANALYSIS_DIR = Path(analysis_paths["analysis"])
    DATA_ROOT = Path(analysis_paths["data"])
    OPTIMIZATION_DIR = Path(analysis_paths["optimization"])

    if MERGE_MODE == "benchmark_points":
        OUTPUT_PATH = OPTIMIZATION_DIR / OUTPUT_FILE
    else:
        output_dir = ANALYSIS_DIR / OUTPUT_FOLDER
        output_dir.mkdir(parents=True, exist_ok=True)
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
        labels = [item.strip() for item in header.split(",") if item.strip()]
    else:
        labels = header.split()

    if len(labels) != len(set(labels)):
        print(f"[WARN] Header has repeated column names in {data_file.name}.")

    return labels


def load_data_file(data_file):
    if not data_file.exists():
        print(f"[WARN] Input file not found, skipped: {data_file}")
        return None, None

    try:
        data = np.loadtxt(data_file, dtype=float, comments="#")
    except Exception as e:
        print(f"[WARN] Could not read {data_file.name}, skipped: {e}")
        return None, None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.size == 0:
        print(f"[WARN] Empty input file, skipped: {data_file}")
        return None, None

    return data, parse_header(data_file)


def fallback_header(n_cols):
    labels = [item.strip() for item in (RAW_HEADER or "").split(",") if item.strip()]
    if len(labels) == n_cols:
        return labels
    return [f"Col_{i}" for i in range(n_cols)]


def headers_match(reference, candidate):
    return reference == candidate


def header_to_text(header):
    if header is None:
        return "<no header>"
    return ", ".join(header)


def has_repeated_columns(header):
    return header is not None and len(header) != len(set(header))


def prepare_data_record(data_file, data, header, reference_header, reference_n_cols):
    n_cols = data.shape[1]

    if header is None:
        print(f"[WARN] No header found in {data_file.name}.")
        if STRICT_HEADER_CHECK:
            print("[ERROR] Merge stopped because STRICT_HEADER_CHECK=True.")
            return None, None, None

        if reference_header is None:
            print(f"[WARN] Using fallback/generic header for first headerless file: {data_file.name}.")
            effective_header = fallback_header(n_cols)
            return effective_header, n_cols, data

        if n_cols != reference_n_cols:
            print(
                f"[WARN] Headerless file {data_file.name} has {n_cols} columns; "
                f"expected {reference_n_cols}. File skipped."
            )
            return None, None, None

        print(f"[WARN] Using reference header for headerless file: {data_file.name}.")
        return list(reference_header), n_cols, data

    if len(header) != n_cols:
        print(f"[WARN] Header mismatch in {data_file.name}: {len(header)} labels vs {n_cols} columns.")
        if STRICT_HEADER_CHECK:
            print("[ERROR] Merge stopped because STRICT_HEADER_CHECK=True.")
            return None, None, None

        if reference_header is None:
            print(f"[WARN] Using fallback/generic header for file with invalid header: {data_file.name}.")
            effective_header = fallback_header(n_cols)
            return effective_header, n_cols, data

        if n_cols != reference_n_cols:
            print(
                f"[WARN] Invalid-header file {data_file.name} has {n_cols} columns; "
                f"expected {reference_n_cols}. File skipped."
            )
            return None, None, None

        print(f"[WARN] Using reference header for file with invalid header: {data_file.name}.")
        return list(reference_header), n_cols, data

    return header, n_cols, data


def print_incompatible_headers(records, level="[ERROR]"):
    print(f"{level} Incompatible headers found:")
    for record in records:
        print(f"  - {record['path'].name}: {header_to_text(record['header'])}")


def common_columns_from_records(records):
    reference_header = records[0]["header"]
    common = []
    for label in reference_header:
        if all(label in record["header"] for record in records):
            common.append(label)
    return common


def merge_common_columns(records, common_header):
    if not common_header:
        print("[ERROR] No common columns found. Merge stopped.")
        return None, None

    if any(has_repeated_columns(record["header"]) for record in records):
        print("[ERROR] Common-column merge requires unique column names in every file.")
        return None, None

    print("[WARN] Headers differ. Merging only common columns because ALLOW_COMMON_COLUMNS_MERGE=True.")
    print("Columns kept: " + ", ".join(common_header))

    projected_blocks = []
    total_rows = 0
    for record in records:
        discarded = [label for label in record["header"] if label not in common_header]
        if discarded:
            print(f"[WARN] {record['path'].name}: discarded columns: {', '.join(discarded)}")
        indices = [record["header"].index(label) for label in common_header]
        projected = record["data"][:, indices]
        projected_blocks.append(projected)
        total_rows += projected.shape[0]
        print(f"  [+] {record['path'].name}: {projected.shape[0]} rows")

    merged = np.vstack(projected_blocks)
    print(f"--> Total rows merged: {total_rows}")
    return merged, common_header


def natural_sort_key(path):
    parts = []
    text = Path(path).name
    current = ""

    for char in text:
        if current and current[-1].isdigit() != char.isdigit():
            parts.append(int(current) if current.isdigit() else current.lower())
            current = char
        else:
            current += char

    if current:
        parts.append(int(current) if current.isdigit() else current.lower())

    return parts


def is_refined_point(path):
    return Path(path).stem.endswith("_R")


def discover_benchmark_files():
    if BENCHMARK_FILES:
        manual_files = [OPTIMIZATION_DIR / file_name for file_name in BENCHMARK_FILES]
        return sorted(manual_files, key=natural_sort_key), []

    found = sorted(OPTIMIZATION_DIR.glob(f"{POINT_PREFIX}_*.dat"), key=natural_sort_key)
    included = []

    for file_path in found:
        refined = is_refined_point(file_path)
        if INCLUDE_ONLY_REFINED and not refined:
            continue
        if not INCLUDE_ONLY_REFINED and EXCLUDE_REFINED and refined:
            continue
        included.append(file_path)

    return included, found


def discover_data_files():
    if INPUT_FILES:
        files = []
        for item in INPUT_FILES:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                data_tag, file_name = item
                files.append(DATA_ROOT / data_tag / file_name)
            else:
                files.append(ANALYSIS_DIR / Path(item))
        source = "manual INPUT_FILES override"
    else:
        input_dir = ANALYSIS_DIR / INPUT_FOLDER
        if not input_dir.exists():
            print(f"[WARN] Input folder not found: {input_dir}")
            return []
        files = list(input_dir.glob(FILE_PATTERN))
        source = f"{INPUT_FOLDER}/{FILE_PATTERN}"

    output_resolved = OUTPUT_PATH.resolve()
    included = [
        path for path in files
        if path.is_file() and path.resolve() != output_resolved
    ]
    included = sorted(included, key=natural_sort_key)

    print(f"Input selection: {source}")
    print(f"Files found: {len(included)}")
    for file_path in included:
        print(f"  - {file_path.name}")

    if not included:
        print("[WARN] No data files matched the configured input selection.")
    return included


def load_benchmark_point(data_file):
    if not data_file.exists():
        print(f"[WARN] Benchmark file not found, skipped: {data_file}")
        return None

    try:
        data = np.loadtxt(data_file, dtype=float, comments="#")
    except Exception as e:
        print(f"[WARN] Could not read {data_file.name}, skipped: {e}")
        return None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.size == 0:
        print(f"[WARN] Empty benchmark file, skipped: {data_file}")
        return None

    return data


def merge_benchmark_points():
    included_files, found_files = discover_benchmark_files()

    print(f"POINT_PREFIX: {POINT_PREFIX}")
    if BENCHMARK_FILES:
        print(f"Files found: manual list in BENCHMARK_FILES ({len(BENCHMARK_FILES)})")
        for file_name in BENCHMARK_FILES:
            print(f"  - {file_name}")
    else:
        print(f"Files found: {len(found_files)}")
        for file_path in found_files:
            print(f"  - {file_path.name}")

    if not included_files:
        if found_files:
            print("[WARN] Benchmark files were found, but none passed the refined-file filters.")
        else:
            print(f"[WARN] No benchmark files found for pattern: {POINT_PREFIX}_*.dat")
        return None

    print(f"Files included: {len(included_files)}")
    merged_blocks = []
    reference_n_cols = None
    total_rows = 0

    for file_path in included_files:
        data = load_benchmark_point(file_path)
        if data is None:
            continue

        n_cols = data.shape[1]
        if reference_n_cols is None:
            reference_n_cols = n_cols
        elif n_cols != reference_n_cols:
            print(f"[WARN] Column count mismatch in {file_path.name}: {n_cols} vs {reference_n_cols}. File skipped.")
            continue

        merged_blocks.append(data)
        total_rows += data.shape[0]
        print(f"  [+] {file_path.name}: {data.shape[0]} rows")

    if not merged_blocks:
        print("[WARN] No compatible benchmark points found.")
        return None

    merged = np.vstack(merged_blocks)
    print(f"Total rows saved: {total_rows}")
    print(f"Output file: {OUTPUT_PATH}")
    return merged


def merge_inputs():
    records = []
    reference_header = None
    reference_n_cols = None

    for data_file in discover_data_files():
        data, header = load_data_file(data_file)
        if data is None:
            continue

        effective_header, n_cols, prepared_data = prepare_data_record(
            data_file, data, header, reference_header, reference_n_cols
        )
        if prepared_data is None:
            if STRICT_HEADER_CHECK:
                return None, None
            continue

        if reference_header is None:
            reference_header = effective_header
            reference_n_cols = n_cols

        records.append({
            "path": data_file,
            "data": prepared_data,
            "header": effective_header,
            "n_cols": n_cols,
        })

    if not records:
        print("[WARN] No compatible input data found.")
        return None, None

    headers_identical = all(headers_match(records[0]["header"], record["header"]) for record in records)
    if headers_identical:
        merged_blocks = []
        total_rows = 0
        for record in records:
            if record["data"].shape[1] != len(records[0]["header"]):
                print(
                    f"[WARN] Column count mismatch in {record['path'].name}: "
                    f"{record['data'].shape[1]} vs {len(records[0]['header'])}."
                )
                if STRICT_HEADER_CHECK:
                    print("[ERROR] Merge stopped because STRICT_HEADER_CHECK=True.")
                    return None, None
                print("[WARN] File skipped.")
                continue

            merged_blocks.append(record["data"])
            total_rows += record["data"].shape[0]
            print(f"  [+] {record['path'].name}: {record['data'].shape[0]} rows")

        if not merged_blocks:
            print("[WARN] No compatible input data found.")
            return None, None

        merged = np.vstack(merged_blocks)
        print(f"--> Total rows merged: {total_rows}")
        return merged, records[0]["header"]

    level = "[ERROR]" if STRICT_HEADER_CHECK or not ALLOW_COMMON_COLUMNS_MERGE else "[WARN]"
    print_incompatible_headers(records, level=level)
    if STRICT_HEADER_CHECK:
        print("[ERROR] Merge stopped because STRICT_HEADER_CHECK=True.")
        return None, None

    if not ALLOW_COMMON_COLUMNS_MERGE:
        print("[ERROR] Merge stopped. Set ALLOW_COMMON_COLUMNS_MERGE=True to merge only common columns.")
        return None, None

    common_header = common_columns_from_records(records)
    return merge_common_columns(records, common_header)


def main():
    configure_analysis()

    print(f"\n=== DATA MERGER ===")
    print(f"Mode: {MERGE_MODE}")
    print(f"Output: {OUTPUT_PATH}")

    if MERGE_MODE == "benchmark_points":
        merged = merge_benchmark_points()
        if merged is None:
            return

        try:
            np.savetxt(OUTPUT_PATH, merged, fmt="%.20e")
            print(f"[SAVE] Benchmark merge complete: {OUTPUT_PATH}")
            print(f"       Rows: {merged.shape[0]}")
            print(f"       Columns: {merged.shape[1]}")
        except OSError as e:
            print(f"[ERROR] Write failed: {e}")
        return

    if MERGE_MODE != "data_files":
        print(f"[ERROR] Unknown MERGE_MODE: {MERGE_MODE}")
        return

    print(f"Input folder: {ANALYSIS_DIR / INPUT_FOLDER}")
    print(f"File pattern: {FILE_PATTERN}")

    merged, header = merge_inputs()
    if merged is None:
        return

    try:
        np.savetxt(OUTPUT_PATH, merged, fmt="%.6e", header=", ".join(header))
        print(f"[SAVE] Merge complete: {OUTPUT_PATH}")
        print(f"       Rows: {merged.shape[0]}")
        print(f"       Columns: {merged.shape[1]}")
    except OSError as e:
        print(f"[ERROR] Write failed: {e}")


if __name__ == "__main__":
    main()
