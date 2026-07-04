# micrOMEGAs backend for ELBAPH2.0

This directory contains the optional micrOMEGAs infrastructure used only when
`"dark_matter"` is included in `ACTIVE_OBSERVABLES`.

## Files

```text
external/micromegas/
|-- build_backend.py
|-- run_micromegas_lib.c
|-- data_run.par
|-- data_DM.par
|-- main_DMsweep.c
`-- libmicromegas.so        # created manually, not tracked by default
```

`run_micromegas_lib.c` is the shared-library wrapper used by the Python backend.
It preserves the legacy microELBAPH input order:

```text
mS, MN1, MN2, MN3, lam2, lam3, lam4, lam5,
yN11, yN12_re, yN12_im, yN13,
yN21, yN22, yN23, yN31, yN32, yN33
```

`main_DMsweep.c` is kept only as an optional standalone legacy C scanner. It is
not part of the ELBAPH2.0 Python scan workflow.

## How To Build libmicromegas.so

ELBAPH2.0 is assumed to live inside the micrOMEGAs model folder:

```text
micromegas/
`-- ModelName/
    `-- ELBAPH2.0/
        `-- external/
            `-- micromegas/
                `-- build_backend.py
```

`build_backend.py` infers the project, model and micrOMEGAs roots from its own
location:

```python
THIS_FILE = Path(__file__).resolve()
ELBAPH_ROOT = THIS_FILE.parents[2]
MICROMEGAS_MODEL_DIR = ELBAPH_ROOT.parent
MICROMEGAS_ROOT = MICROMEGAS_MODEL_DIR.parent
```

No absolute paths need to be hardcoded. Make sure the model directory has the
micrOMEGAs build products expected by the linker and a suitable `data_run.par`,
then run the builder manually:

```bash
python external/micromegas/build_backend.py
```

The generated library is copied to:

```text
external/micromegas/libmicromegas.so
```

The builder may call `gcc` because its only purpose is compilation. ELBAPH2.0
does not call this builder automatically from `likelihood.py`,
`obs_dark_matter.py`, scanners or optimizers.

## Runtime Configuration

In the Analysis settings, point to the compiled library:

```python
MICROMEGAS_LIB = "external/micromegas/libmicromegas.so"
```

If `"dark_matter"` is not in `ACTIVE_OBSERVABLES`, this file is not needed and
micrOMEGAs is not imported or loaded.

If `"dark_matter"` is active and the library is missing, the backend returns an
explicit error and `obs_dark_matter.py` applies an explicit DM penalty. No dummy
values such as `Omega = 0.12` or `sigmaSI = 0` are used silently.
