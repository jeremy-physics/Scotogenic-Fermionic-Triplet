# ELBAPH2.0 Evolutionary Likelihood BSM Analysis for PHenomenology

ELBAPH2.0 is a personal tool for model-specific BSM phenomenology studies. It organizes fixed physics configurations, numerical tasks, and plotting data without attempting to be a model-independent public framework.

The workflow is built around three concepts:

- **Analysis:** a fixed phenomenological setup: model, observables, bounds, targets, penalties, and output configuration.
- **Task:** a concrete execution of a script within the active Analysis.
- **PlotSet:** an independent folder or configuration for plotting prepared scan or output data.

## Project structure

```text
settings/                       Analysis selection and configuration
  current_analysis.py           Selects the active Analysis
  analyses/                     Analysis files and the base template
scripts/                        Executable workflow tasks
src/                            Model calculations and shared infrastructure
  variables.py                  Canonical variable names and variable sets
  targets.py                    Experimental targets and chi-square sets
  bounds.py                     Bounds and scan/physical-space transformations
  likelihood.py                 Model evaluation and the values dictionary
  penalties.py                  Non-statistical restrictions
  backends/                     Optional external-library adapters
external/micromegas/            Optional micrOMEGAs build infrastructure
outputs/                        Analysis outputs and independent PlotSets
```

## Core design

- `variables.py` defines canonical variable names and reusable variable sets.
- `likelihood.py` computes the model and fills the shared `values` dictionary.
- `targets.py` connects entries in `values` to experimental targets used by the chi-square.
- `bounds.py` defines scan bounds and the scan-space/physical-space transformation.
- `penalties.py` defines additional non-statistical restrictions.
- `src/penalties.py` also provides `hybrid_pen`, a soft guide penalty for steering scans toward preferred regions without imposing a hard bound.
- `scripts/` executes tasks using the active Analysis configuration.

Variables are named in `variables.py`, but their numerical values are assigned in `likelihood.py` through the `values` dictionary.

## User/agent workflow contract

ELBAPH2.0 is designed so that a human user and an AI coding agent can work on the same repository without losing transparency.

The intended division is:

- **Analysis files** in `settings/analyses/` define the phenomenological setup: active observables, target sets, custom bounds, raw/show variables and output roots.
- **Current analysis selection** lives in `settings/current_analysis.py`.
- **Task scripts** in `scripts/` execute concrete jobs. Routine task changes should be limited to constants inside `USER/AGENT CONFIGURATION` blocks.
- **Model definitions** in `src/model_definitions.py` contain explicit model-dependent physics formulae, such as mass matrices.
- **Likelihood value construction** in `src/likelihood.py::_build_values` defines how model-dependent quantities are assembled into the canonical `values` dictionary used by targets, penalties, raw outputs and viewers.
- **Core infrastructure** in `src/` defines reusable machinery such as likelihood evaluation, bounds transformations, targets, variables, penalties and backends.
- **PlotSets** in `outputs/plots/<AnalysisName>/<PlotSetName>/` contain analysis-specific plotting scripts and generated PDF figures.

The AI agent should not create hidden analyses, hidden scripts, alternative private execution paths or selector functions to avoid editing the intended file.

If two physical model variants must be compared, for example two different neutrino matrices, each variant should be run from an explicit source-code state. The relevant function in `src/model_definitions.py` should be edited directly for each case. If the variant requires changing how the `values` dictionary is built, `src/likelihood.py::_build_values` may also be edited directly. The source snapshot stored in each Analysis output preserves the source state used for that run.

A model-selection interface should only be introduced if the user explicitly requests it.

For routine execution, scripts should expose editable constants inside:

```python
# === USER/AGENT CONFIGURATION START ===
...
# === USER/AGENT CONFIGURATION END ===
```

The agent may edit those blocks to run tasks. Changes outside those blocks should be treated as code changes and reported explicitly.

## Analysis files

An Analysis lives at:

```text
settings/analyses/<analysis_name>.py
```

The clean distribution contains only `settings/analyses/analysis_base.py` as a template and has no active Analysis by default. Create a new Analysis with `python scripts/create_analysis.py`; the selected Analysis is then recorded in `settings/current_analysis.py`. Important settings include:

```python
ACTIVE_OBSERVABLES
ACTIVE_CHI_TARGET_SETS
CUSTOM_CHI_TARGETS
CUSTOM_BOUNDS
RAW_VARIABLE_SETS
SHOW_VARIABLE_SETS
EXTRA_RAW_VARIABLES
EXTRA_SHOW_VARIABLES
```

Recommended defaults are:

```python
ACTIVE_CHI_TARGET_SETS = None

RAW_VARIABLE_SETS = [
    "active_chi_targets",
    "cost_breakdown",
]

SHOW_VARIABLE_SETS = [
    "active_chi_targets",
    "cost_breakdown",
]
```

`active_chi_targets` is resolved dynamically to the canonical model variables entering the active chi-square. `all_model_variables` remains available for debugging, but it is not recommended as a default because inactive observable blocks may produce `nan` values.

## Python environment

A minimal Python environment can be installed with:

```bash
pip install -r requirements.txt
```

The required Python packages are NumPy, SciPy and Matplotlib. The micrOMEGAs backend is optional and requires an external micrOMEGAs build; it is not installed through `requirements.txt`.

## Basic workflow

Edit the user variables at the beginning of each script, then run tasks directly:

```bash
python scripts/create_analysis.py
python scripts/validate_analysis.py
python scripts/run_result_viewer.py
python scripts/run_DE_single.py
python scripts/run_DE_sweep.py
python scripts/run_global_scan.py
python scripts/run_analysis_sampler.py
python scripts/run_analysis_stats.py
python scripts/run_merge.py
```

- `create_analysis.py`: creates an Analysis from `analysis_base.py` and can select it as active.
- `validate_analysis.py`: checks settings, bounds, targets, raw/show variables, micrOMEGAs configuration, and an available benchmark.
- `run_result_viewer.py`: evaluates and reports one optimization point.
- `run_DE_single.py`: refines one existing point with Differential Evolution.
- `run_DE_sweep.py`: searches for multiple optimization points.
- `run_global_scan.py`: generates a global scan dataset.
- `run_analysis_sampler.py`: perturbs a benchmark locally and saves valid rows.
- `run_analysis_stats.py`: calculates summary statistics and correlations from a dataset.
- `run_merge.py`: combines optimization points or compatible data files.

Long-running `run_*.py` tasks may be interrupted manually from the terminal. Very large iteration limits in `run_DE_single.py` are deliberate and should not be reduced unless explicitly requested; after an interruption, inspect and report any partial outputs before deciding how to continue.

## Result viewer

Run:

```bash
python scripts/run_result_viewer.py
```

The viewer reports variables selected by `SHOW_VARIABLE_SETS`, relative errors for active targets, the cost breakdown, and consistency checks. Optional Mathematica and micrOMEGAs exports are controlled by editable flags such as:

```python
PRINT_RELATIVE_ERROR
PRINT_MATH
PRINT_MICRO
```

## micrOMEGAs backend

The intended deployment places ELBAPH2.0 inside a micrOMEGAs model directory:

```text
micromegas/
└── ModelName/
    └── ELBAPH2.0/
```

The builder derives the ELBAPH project root from its own location. With the expected layout, the parent of ELBAPH2.0 is the micrOMEGAs model directory and its parent is the micrOMEGAs installation root; no placeholder paths are required.

Build with:

```bash
python external/micromegas/build_backend.py
```

The expected library is:

```text
external/micromegas/libmicromegas.so
```

A dark-matter Analysis uses:

```python
MICROMEGAS_LIB = "external/micromegas/libmicromegas.so"

ACTIVE_OBSERVABLES = [
    "quarks",
    "leptons",
    "scalars",
    "dark_matter",
]
```

micrOMEGAs is optional and is not loaded when `dark_matter` is inactive.

## Outputs

Each prepared Analysis uses:

```text
outputs/analyses/<AnalysisName>/
├── optimization/
├── data/
│   ├── scans/
│   ├── stats/
│   └── merged/
├── logs/
├── settings_snapshot/
└── source_snapshot/
```

Optimization points are stored in scan-space. Dataset headers are generated from the resolved `RAW_VARIABLE_SETS`, so column order follows the Analysis configuration.

Plot data, Analysis-specific plot scripts, and figures can be kept independently under `outputs/plots/<AnalysisName>/<PlotSetName>/`.

## Plot workflow

Plots are organized as PlotSets under:

```text
outputs/plots/<AnalysisName>/<PlotSetName>/
```

A plotting task is complete only when it leaves both:

```text
plot_script.py
plot_output.pdf
```

The generated PDF is the visual result, while the Python script is the reproducible and editable source used to create it.

The preferred style for AI-generated plot scripts is direct, explicit Matplotlib code. The files in `src/plot_tools.py`, the style notebooks and the `.mplstyle` files should be treated as visual references for the user's plotting grammar. They are not mandatory plotting APIs.

By default, AI-generated scripts should not hide the plot construction inside compact helper functions such as `sp_scatter`, `cb_scatter`, `del_reg` or similar wrappers. The user should be able to open the script and directly edit the scatter calls, lines, shaded regions, annotations, axes, legend and export options.

Generated plot scripts should follow this structure:

```text
1. global plot configuration
2. data loading and preparation
3. global style
4. figure and axes
5. main plots
6. lines and regions
7. texts, annotations and benchmark markers
8. axes and legend
9. PDF export
```

Plots should be exported as PDF. PNG export is not part of the standard ELBAPH2.0 plotting workflow unless explicitly requested.

## Cost function

- `chi`: statistical contribution from active chi targets.
- `pen_structure`: internal structural penalties.
- `pen_restriction`: additional phenomenological restrictions.
- `penalty_total`: `pen_structure + pen_restriction`.
- `cost_function`: `chi + penalty_total`.

The optimizers minimize `cost_function`.

## Plotting style

The default plotting style follows the user's paper-style matplotlib setup. Styles live in `styles/`, reusable plotting helpers live in `src/plot_tools.py`, and `scripts/plot_template.py` is the recommended starting point for new plots.

The visual grammar uses serif fonts, LaTeX labels, inward ticks, visible minor ticks, compact colorbars, shaded excluded regions, PDF output, and configurable marker edges. The helpers are optional: complex plots may use direct matplotlib code while preserving the same visual language.

Variable labels for plots can be centralized in `src/variables.py` through `VARIABLE_LABELS`. Plot scripts should use these labels when available. The preferred primary plotting colors are `c1`, `c2` and `c3`, while `c_reg*` and `c_esc*` are intended for regions and region annotations.

## Current status

ELBAPH2.0 is currently model-specific and intended for personal research workflows, not as a model-independent public package. The architecture is stable enough for operational use, while individual Analysis files remain user-defined.
