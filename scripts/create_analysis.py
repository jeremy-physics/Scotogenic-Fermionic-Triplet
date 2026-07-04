import os
import re
import sys
from pathlib import Path


# === USER/AGENT CONFIGURATION START ===

NEW_ANALYSIS_NAME = "analysis_dd"
TEMPLATE_ANALYSIS = "test"

ACTIVE_OBSERVABLES = [
    "leptons",
    "scalars",
    "direct_detection",
]

DEFAULT_ANALYSIS_LIBRARY_SETTINGS = """
ACTIVE_CHI_TARGET_SETS = None
CHI_SIGMA_SCALE = 3.0
CUSTOM_CHI_TARGETS = {}

CUSTOM_BOUNDS = {}

RAW_VARIABLE_SETS = [
    "active_chi_targets",
    "cost_breakdown",
]
SHOW_VARIABLE_SETS = [
    "active_chi_targets",
    "cost_breakdown",
]
EXTRA_RAW_VARIABLES = []
EXTRA_SHOW_VARIABLES = []
""".strip()

SET_AS_CURRENT = True
OVERWRITE_EXISTING = False

# === USER/AGENT CONFIGURATION END ===


# --- Path Configuration ---
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = CURRENT_DIR.parent
PROJECT_SRC = PROJECT_ROOT / "src"

module_path = str(PROJECT_SRC)
if module_path not in sys.path:
    sys.path.append(module_path)

from io_tools import prepare_analysis


ANALYSES_DIR = PROJECT_ROOT / "settings" / "analyses"
CURRENT_ANALYSIS_FILE = PROJECT_ROOT / "settings" / "current_analysis.py"


def validate_analysis_name(name):
    if not re.match(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$", name):
        raise ValueError(
            "Analysis names may contain only letters, numbers, underscores and hyphens, "
            "and must not start with a hyphen."
        )


def format_active_observables(observables):
    lines = ["ACTIVE_OBSERVABLES = ["]
    for observable in observables:
        lines.append(f'    "{observable}",')
    lines.append("]")
    return "\n".join(lines)


def replace_or_insert_assignment(text, name, value):
    assignment = f'{name} = "{value}"'
    pattern = rf"^{name}\s*=.*$"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, assignment, text, count=1, flags=re.MULTILINE)
    return assignment + "\n\n" + text


def replace_or_insert_active_observables(text, observables):
    block = format_active_observables(observables)
    pattern = r"^ACTIVE_OBSERVABLES\s*=\s*\[.*?\]\s*"
    if re.search(pattern, text, flags=re.MULTILINE | re.DOTALL):
        return re.sub(pattern, block + "\n\n", text, count=1, flags=re.MULTILINE | re.DOTALL)

    analysis_pattern = r"^ANALYSIS_NAME\s*=.*$"
    match = re.search(analysis_pattern, text, flags=re.MULTILINE)
    if match:
        insert_at = match.end()
        return text[:insert_at] + "\n\n" + block + text[insert_at:]

    return block + "\n\n" + text


def ensure_analysis_library_settings(text):
    required_keys = [
        "ACTIVE_CHI_TARGET_SETS",
        "CHI_SIGMA_SCALE",
        "CUSTOM_CHI_TARGETS",
        "CUSTOM_BOUNDS",
        "RAW_VARIABLE_SETS",
        "SHOW_VARIABLE_SETS",
        "EXTRA_RAW_VARIABLES",
        "EXTRA_SHOW_VARIABLES",
    ]
    if all(re.search(rf"^{key}\s*=", text, flags=re.MULTILINE) for key in required_keys):
        return text

    observable_pattern = r"^ACTIVE_OBSERVABLES\s*=\s*\[.*?\]\s*"
    match = re.search(observable_pattern, text, flags=re.MULTILINE | re.DOTALL)
    if match:
        insert_at = match.end()
        return text[:insert_at] + "\n" + DEFAULT_ANALYSIS_LIBRARY_SETTINGS + "\n\n" + text[insert_at:]

    return DEFAULT_ANALYSIS_LIBRARY_SETTINGS + "\n\n" + text


def write_current_analysis(analysis_name):
    content = (
        f'ACTIVE_ANALYSIS = "{analysis_name}"\n'
        'SETTINGS_FILE = f"settings/analyses/{ACTIVE_ANALYSIS}.py"\n'
    )
    CURRENT_ANALYSIS_FILE.write_text(content, encoding="utf-8")
    print(f"[SAVE] Current analysis set to: {analysis_name}")


def create_settings_file(template_path, output_path):
    text = template_path.read_text(encoding="utf-8")
    text = replace_or_insert_assignment(text, "ANALYSIS_NAME", NEW_ANALYSIS_NAME)
    text = replace_or_insert_active_observables(text, ACTIVE_OBSERVABLES)
    text = ensure_analysis_library_settings(text)
    output_path.write_text(text, encoding="utf-8")
    print(f"[SAVE] Created settings file: {output_path}")


def main():
    validate_analysis_name(NEW_ANALYSIS_NAME)
    validate_analysis_name(TEMPLATE_ANALYSIS)

    template_path = ANALYSES_DIR / f"{TEMPLATE_ANALYSIS}.py"
    output_path = ANALYSES_DIR / f"{NEW_ANALYSIS_NAME}.py"

    if not template_path.exists():
        print(f"[ERROR] Template analysis not found: {template_path}")
        return

    if output_path.exists() and not OVERWRITE_EXISTING:
        print(f"[WARN] Analysis already exists and OVERWRITE_EXISTING=False: {output_path}")
        print("[WARN] Existing settings file was not modified.")
    else:
        create_settings_file(template_path, output_path)

    if SET_AS_CURRENT:
        write_current_analysis(NEW_ANALYSIS_NAME)

    if not output_path.exists():
        print(f"[ERROR] Settings file is not available after creation step: {output_path}")
        return

    settings_file = f"settings/analyses/{NEW_ANALYSIS_NAME}.py"
    try:
        settings, paths = prepare_analysis(settings_file)
    except Exception as exc:
        print(f"[ERROR] Could not prepare Analysis folders: {exc}")
        return

    print(f"[OK] Analysis ready: {settings.ANALYSIS_NAME}")
    print(f"[OK] Output folder: {paths['analysis']}")


if __name__ == "__main__":
    main()
