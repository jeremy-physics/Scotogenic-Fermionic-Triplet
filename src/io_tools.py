from importlib import util
from pathlib import Path
from shutil import copy2
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SUBDIRS = (
    "settings_snapshot",
    "source_snapshot",
    "optimization",
    "data",
    "data/scans",
    "data/stats",
    "data/merged",
    "logs",
)
SETTINGS_SNAPSHOT_FILES = (
    "src/constants.py",
    ("src/bounds.py",),
    ("src/targets.py",),
)
SOURCE_SNAPSHOT_FILES = (
    "src/likelihood.py",
    "src/model_definitions.py",
    "src/obs_quarks.py",
    "src/obs_leptons.py",
    "src/obs_scalars.py",
    "src/variables.py",
    "src/targets.py",
    "src/penalties.py",
    "src/obs_dark_matter.py",
    "src/backends/micromegas_backend.py",
)


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_settings(settings_file):
    settings_path = resolve_project_path(settings_file)

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    if settings_path.suffix != ".py":
        raise ValueError(f"Settings file must be a .py file: {settings_path}")

    module_name = f"elbaph_settings_{settings_path.stem}"
    spec = util.spec_from_file_location(module_name, settings_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load settings file: {settings_path}")

    settings = util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(settings)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    settings.SETTINGS_FILE = str(settings_path)
    return settings


def get_settings_file(use_current_analysis=True, local_settings_file=None):
    if not use_current_analysis:
        if not local_settings_file:
            raise ValueError("LOCAL_SETTINGS_FILE must be set when USE_CURRENT_ANALYSIS=False.")
        return local_settings_file

    current_path = resolve_project_path("settings/current_analysis.py")
    if not current_path.exists():
        raise FileNotFoundError(f"Current analysis selector not found: {current_path}")

    spec = util.spec_from_file_location("elbaph_current_analysis", current_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load current analysis selector: {current_path}")

    current = util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(current)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode

    settings_file = getattr(current, "SETTINGS_FILE", None)
    if not settings_file:
        raise SystemExit(
            "No active analysis selected. Create one with "
            "scripts/create_analysis.py or set ACTIVE_ANALYSIS in "
            "settings/current_analysis.py."
        )
    return settings_file


def get_analysis_dir(settings):
    return resolve_project_path(settings.OUTPUT_ROOT) / settings.ANALYSIS_NAME


def ensure_analysis_folders(settings):
    analysis_dir = get_analysis_dir(settings)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    paths = {"analysis": analysis_dir}
    for subdir in ANALYSIS_SUBDIRS:
        subdir_path = analysis_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

        gitkeep = subdir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

        paths[subdir.replace("/", "_")] = subdir_path

    notes_file = analysis_dir / "notes.txt"
    if not notes_file.exists():
        notes_file.touch()
    paths["notes"] = notes_file

    return paths


def copy_snapshot_file(source_file, snapshot_dir):
    if isinstance(source_file, (tuple, list)):
        for candidate in source_file:
            source_path = resolve_project_path(candidate)
            if source_path.exists():
                return copy_snapshot_file(candidate, snapshot_dir)
        candidates = ", ".join(str(resolve_project_path(candidate)) for candidate in source_file)
        print(f"[WARN] Snapshot skipped; no candidate file found: {candidates}")
        return None

    source_path = resolve_project_path(source_file)
    if not source_path.exists():
        print(f"[WARN] Snapshot skipped; file not found: {source_path}")
        return None

    destination = Path(snapshot_dir) / source_path.name
    if source_path.resolve() == destination.resolve():
        return destination

    copy2(source_path, destination)
    return destination


def save_analysis_snapshots(settings, paths=None):
    if paths is None:
        paths = ensure_analysis_folders(settings)

    copied = {
        "settings_snapshot": [],
        "source_snapshot": [],
    }

    settings_file = getattr(settings, "SETTINGS_FILE", None)
    if settings_file:
        copied_file = copy_snapshot_file(settings_file, paths["settings_snapshot"])
        if copied_file is not None:
            copied["settings_snapshot"].append(copied_file)
    else:
        print("[WARN] Snapshot skipped; settings file path is not available.")

    for source_file in SETTINGS_SNAPSHOT_FILES:
        copied_file = copy_snapshot_file(source_file, paths["settings_snapshot"])
        if copied_file is not None:
            copied["settings_snapshot"].append(copied_file)

    if getattr(settings, "SAVE_SOURCE_SNAPSHOT", False):
        for source_file in SOURCE_SNAPSHOT_FILES:
            copied_file = copy_snapshot_file(source_file, paths["source_snapshot"])
            if copied_file is not None:
                copied["source_snapshot"].append(copied_file)

    return copied


def publish_settings_environment(settings):
    active = getattr(settings, "ACTIVE_OBSERVABLES", [])
    os.environ["ELBAPH_ACTIVE_OBSERVABLES"] = ",".join(str(item) for item in active)

    micromegas_lib = getattr(settings, "MICROMEGAS_LIB", None)
    if micromegas_lib:
        os.environ["ELBAPH_MICROMEGAS_LIB"] = str(resolve_project_path(micromegas_lib))
    else:
        os.environ.pop("ELBAPH_MICROMEGAS_LIB", None)


def configure_loaded_likelihood(settings):
    likelihood = sys.modules.get("likelihood")
    configure = getattr(likelihood, "configure_from_settings", None) if likelihood else None
    if configure is not None:
        configure(settings)


def prepare_analysis(settings_file):
    settings = load_settings(settings_file)
    publish_settings_environment(settings)
    configure_loaded_likelihood(settings)
    paths = ensure_analysis_folders(settings)
    save_analysis_snapshots(settings, paths)
    return settings, paths
