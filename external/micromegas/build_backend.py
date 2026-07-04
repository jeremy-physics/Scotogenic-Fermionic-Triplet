import shutil
import subprocess
from pathlib import Path


# Paths are inferred assuming ELBAPH2.0 lives inside the micrOMEGAs model directory.
THIS_FILE = Path(__file__).resolve()
ELBAPH_ROOT = THIS_FILE.parents[2]
MICROMEGAS_MODEL_DIR = ELBAPH_ROOT.parent
MICROMEGAS_ROOT = MICROMEGAS_MODEL_DIR.parent

OUTPUT_LIB = Path("external/micromegas/libmicromegas.so")
WRAPPER_C = Path("external/micromegas/run_micromegas_lib.c")

# Relative layout expected when compiling inside MICROMEGAS_MODEL_DIR.
CALCHEP = MICROMEGAS_ROOT / "CalcHEP_src"

CFLAGS = ["-shared", "-g", "-fsigned-char", "-std=gnu99", "-fPIC", "-fopenmp"]
LDFLAGS = ["-rdynamic", "-ldl", "-lm", "-lpthread"]
INCLUDES = ["-I../include"]
LINK_LIBS = [
    "lib/aLib.a",
    "../lib/micromegas.a",
    "work/work_aux.a",
    str(CALCHEP / "lib/dynamic_me.a"),
    str(CALCHEP / "lib/sqme_aux.so"),
    str(CALCHEP / "lib/libSLHAplus.a"),
    str(CALCHEP / "lib/num_c.a"),
    str(CALCHEP / "lib/serv.a"),
    str(CALCHEP / "lib/ntools.a"),
    str(CALCHEP / "../lib/maxGap.so"),
    str(CALCHEP / "lib/lhapdf.so"),
    "../lib/dummy.a",
    str(CALCHEP / "lib/dummy.a"),
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def compile_backend():
    wrapper_source = resolve_project_path(WRAPPER_C)
    output_library = resolve_project_path(OUTPUT_LIB)
    model_dir = Path(MICROMEGAS_MODEL_DIR)

    print("=" * 40)
    print("  ELBAPH2.0 MICROMEGAS BACKEND BUILDER")
    print("=" * 40)
    print(f"micrOMEGAs root:  {MICROMEGAS_ROOT}")
    print(f"Model directory:  {model_dir}")
    print(f"Wrapper C:        {wrapper_source}")
    print(f"Output library:   {output_library}")

    if not wrapper_source.exists():
        print(f"[ERROR] Wrapper C file not found: {wrapper_source}")
        return False
    if not model_dir.exists():
        print(f"[ERROR] MICROMEGAS_MODEL_DIR not found: {model_dir}")
        return False

    staged_wrapper = model_dir / wrapper_source.name
    shutil.copy2(wrapper_source, staged_wrapper)

    cmd = (
        ["gcc"]
        + CFLAGS
        + ["-o", output_library.name, staged_wrapper.name]
        + INCLUDES
        + LINK_LIBS
        + LDFLAGS
    )

    try:
        subprocess.run(cmd, cwd=model_dir, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Compilation failed with exit code {exc.returncode}.")
        return False

    built_library = model_dir / output_library.name
    if not built_library.exists():
        print(f"[ERROR] Compiler finished but library was not created: {built_library}")
        return False

    output_library.parent.mkdir(parents=True, exist_ok=True)
    if built_library.resolve() != output_library.resolve():
        shutil.copy2(built_library, output_library)

    print(f"[OK] Library available at: {output_library}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if compile_backend() else 1)
