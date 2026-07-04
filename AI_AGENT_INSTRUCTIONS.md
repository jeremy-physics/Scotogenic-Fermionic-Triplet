ELBAPH2.0 should be treated as a stable working architecture. The AI agent should primarily use the architecture to define Analysis files, configure tasks, run scripts, inspect outputs and generate plots. The AI agent should not redesign the architecture unless the requested analysis genuinely requires new functionality. In that case, the AI agent must explain the required change and ask the user before modifying core files.

## Operational role

The AI agent may normally:

- create new Analysis files from `analysis_base.py`;
- edit `ACTIVE_OBSERVABLES`;
- edit `CUSTOM_BOUNDS`;
- edit `CUSTOM_CHI_TARGETS`;
- edit `RAW_VARIABLE_SETS` and `SHOW_VARIABLE_SETS`;
- run `scripts/validate_analysis.py`;
- run `scripts/run_result_viewer.py`;
- run optimization, scan, sampler, stats and merge scripts when asked;
- inspect outputs and file headers;
- create plotting scripts inside the corresponding PlotSet folder;
- generate PDF plots from existing scan, sampler or statistics files;
- report results clearly.

The AI agent should use the existing architecture for routine work and should not redesign its central components unless the user explicitly approves a necessary architectural change.

## Agent edit contract

ELBAPH2.0 is designed for transparent collaboration between a human user and an AI coding agent. The AI agent must prefer explicit, inspectable edits over hidden abstractions or private convenience layers.

The agent must distinguish between five edit classes:

1. **Analysis configuration edits.** These are normal edits to files in `settings/analyses/` and to `settings/current_analysis.py`. They are used to select active observables, target sets, custom bounds, raw/show variables, output roots and analysis-level options.

2. **Routine task execution edits.** These are edits to user-facing constants in executable scripts under `scripts/`. For routine execution, the agent may edit only inside blocks marked:

```python
# === USER/AGENT CONFIGURATION START ===
...
# === USER/AGENT CONFIGURATION END ===
```

The agent must not change imports, internal functions, control flow, optimization logic, scan logic, likelihood calls, output formats or hidden defaults unless the user explicitly requests a code change.

3. **Model-definition edits.** These are direct edits to model-dependent physics definitions. The main files are `src/model_definitions.py` and, when required, `src/likelihood.py::_build_values`. These edits are allowed when the user asks for a physical model change or comparison of physical variants.

4. **Penalty edits.** These are direct edits to `src/penalties.py` or related penalty machinery. Penalties must be implemented in a transparent and auditable style, with explicit variables, explicit conditions and simple algebra.

5. **Architecture edits.** These modify reusable infrastructure, such as bounds transformations, target machinery, variable registries, backends, likelihood control flow, snapshot logic or output infrastructure. The agent must not perform architecture edits as part of routine execution. They require an explicit user request.

The AI agent must not optimize for its own convenience by creating hidden selectors, private registries, hidden analysis files, temporary persistent scripts, notebooks, wrappers or alternative execution paths that the user did not request.

Persistent files must be placed in documented project locations and reported at the end of the task. One-off shell commands or inline Python commands are acceptable for inspection, provided they do not leave hidden persistent artifacts.

## Files the AI agent may normally edit

For a specific analysis, the AI agent may edit:

```text
settings/current_analysis.py
settings/analyses/<analysis_name>.py
```

When the user requests a specific task, the AI agent may also edit the local configuration variables at the beginning of existing scripts such as:

```text
scripts/run_result_viewer.py
scripts/run_global_scan.py
scripts/run_DE_scan.py
scripts/run_analysis_sampler.py
scripts/run_analysis_stats.py
scripts/run_merge.py
```

These changes should remain operational and limited to items such as:

- number of points;
- input and output filenames;
- Analysis name;
- input and output folders;
- printing flags;
- variables to display;
- diagnostic thresholds.

The AI agent should not rewrite the central logic of these scripts unless the user explicitly requests it.

## Routine script execution

For routine tasks, executable scripts in `scripts/` are controlled through explicit configuration blocks.

The agent may edit values inside:

```python
# === USER/AGENT CONFIGURATION START ===
...
# === USER/AGENT CONFIGURATION END ===
```

Typical allowed routine edits include:

- input file names;
- output file names;
- number of scan points;
- seeds;
- thresholds;
- overwrite flags;
- PlotSet names;
- analysis names;
- small execution parameters exposed as constants.

For routine execution, the agent must not edit code outside the configuration block. In particular, it must not add helper functions, change imports, alter the optimization algorithm, change the likelihood call, change multiprocessing behavior, change output formats or introduce new control flow.

If a requested task cannot be completed by editing only the configuration block, the agent must treat it as a code-change task and report exactly which file and logic need to be changed.

## Core architecture files

The following files define the architecture or computational physics:

```text
src/variables.py
src/targets.py
src/bounds.py
src/likelihood.py
src/penalties.py
src/model_definitions.py
src/obs_leptons.py
src/obs_quarks.py
src/obs_scalars.py
src/obs_dark_matter.py
src/backends/micromegas_backend.py
external/micromegas/
```

The AI agent should not modify these files for routine usage. If a requested analysis requires changing one of them, the AI agent must first explain why the change is necessary, which file and function would be modified, and what validation would be performed. It must then ask the user for approval.

As an exception, `src/variables.py`, `src/targets.py`, `src/bounds.py` and `src/penalties.py` may be edited when the user explicitly requests adding a variable, target, bound, penalty, variable set, label or canonical name. The AI agent should avoid duplicated lists and preserve the single canonical source for names.

## Variables and values

- `src/variables.py` defines canonical names and variable sets.
- `src/likelihood.py` computes numerical values and fills `values[name]`.
- `src/targets.py` connects `values[name]` to experimental targets through `model_key`.
- Scripts should not manually reconstruct observables.

If a new output column is needed, add it to a variable set or use `EXTRA_RAW_VARIABLES` or `EXTRA_SHOW_VARIABLES`. If a new chi-square observable is needed, add it to `src/targets.py`. If a new computed value is needed, it must be filled in `values` by `src/likelihood.py`, but the AI agent must ask before modifying `src/likelihood.py` unless the user explicitly requested that change.

## Model-variant workflow

When the user asks to study alternative physical definitions, such as two different neutrino mass matrices, the agent must treat each alternative as a concrete source-code state.

Correct workflow:

1. Name the variants clearly, for example `case_A` and `case_B`.
2. For `case_A`, directly edit the relevant physics definition in `src/model_definitions.py`.
3. If the physical variant requires changing how model-dependent quantities are assembled into the canonical `values` dictionary, directly edit `src/likelihood.py::_build_values`.
4. Validate the source state.
5. Run the requested task.
6. Rely on the Analysis source snapshot to preserve the exact code state used for the run.
7. Repeat the process explicitly for `case_B`.

The agent must not replace direct model edits with hidden selectors, dispatch functions, registries, wrappers or analysis-name conditionals unless the user explicitly requests a multi-model interface.

Forbidden by default:

```python
def choose_neutrino_matrix(case):
    ...
```

```python
if MATRIX_CASE == "A":
    ...
elif MATRIX_CASE == "B":
    ...
```

```python
if ACTIVE_ANALYSIS == "case_A":
    ...
```

```python
matrix_registry = {
    "case_A": ...,
    "case_B": ...,
}
```

Analysis files should configure analyses. They should not be used to smuggle alternative physics definitions into the model unless the user explicitly requests that design.

For model variants, the preferred principle is:

```text
physical variant -> explicit source-code state
```

not:

```text
physical variant -> hidden selector or private agent interface
```

## Penalty implementation style

Penalties must be written in a transparent and auditable style. Prefer explicit variables, explicit conditions and simple algebra over compact abstractions.

A new penalty should normally follow this form:

```python
penalty_name = 0.0

if not condition:
    violation = ...
    penalty_name = coefficient * violation ** 2
```

The corresponding mathematical structure should be directly readable as:

```text
P_i = 0                         if the constraint is satisfied
P_i = coefficient * Delta_i^2   if the constraint is violated
```

The agent must not introduce helper functions, vectorized tricks, dispatch tables, smooth hidden barriers, compact transformations or non-obvious penalty frameworks unless the user explicitly requests them.

When adding or changing a penalty, the agent must report:

- the physical condition being imposed;
- the mathematical form of the penalty;
- the coefficient used;
- the exact key added to the penalties dictionary;
- a minimal validation or sanity check.

## Analysis workflow

The clean repository contains only `settings/analyses/analysis_base.py` as a template and has no active Analysis by default. Create a new Analysis with `python scripts/create_analysis.py` before validation or execution tasks.

The recommended Analysis defaults are:

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

`active_chi_targets` is a dynamic set containing the model variables entering the active chi-square. `all_model_variables` is useful for debugging, but is not recommended as a default because inactive blocks may produce `NaN` values.

Before long scans, the AI agent should run:

```bash
python scripts/validate_analysis.py
python scripts/run_result_viewer.py
```

After these checks pass, the AI agent should use a small scan when practical before starting a long scan.

## Interrupting long runs

The `run_*.py` scripts may be interrupted from the terminal when necessary. This applies especially to:

```text
scripts/run_DE_single.py
scripts/run_DE_sweep.py
scripts/run_global_scan.py
scripts/run_DE_scan.py
scripts/run_analysis_sampler.py
```

Some optimization scripts intentionally use very large iteration limits. This is deliberate, because difficult model fits may require very long runs, sometimes lasting days. The AI agent must not reduce `maxiter` or hard-code shorter stopping criteria unless the user explicitly asks for it.

If a run appears stuck in a valley, the user may interrupt it manually from the terminal. The AI agent should treat such an interruption as part of the workflow, not necessarily as a program failure. After an interruption, the AI agent should inspect any partial outputs, avoid assuming that the entire run failed, report clearly which files exist, and continue from those results if the user requests it.

## Plotting workflow

ELBAPH2.0 uses the user's plotting aesthetic through:

```text
styles/paper_style.mplstyle
styles/paper_style_colorbar.mplstyle
src/plot_tools.py
```

The AI agent should use this style as a visual guide, not as a rigid framework. For plotting tasks, it should normally use direct, explicit Matplotlib code while preserving the visual grammar:

- serif fonts;
- LaTeX labels;
- inward ticks;
- visible minor ticks;
- compact colorbars;
- PDF output;
- configurable marker edges;
- shaded or hatched excluded regions when needed;
- no Seaborn.

For axis labels, the AI agent should first try `VARIABLE_LABELS` through `get_variable_label` from `src/variables.py`. If a label is not defined, it may use the raw variable name. The AI agent should not hardcode large label dictionaries inside individual plot scripts when the labels already exist in `src/variables.py`.

Color preferences:

- The user's preferred primary colors are `c1`, `c2` and `c3`.
- `c0` may be used when needed, but `c1`, `c2` and `c3` are preferred for main datasets.
- `c_reg*` colors are preferred for excluded or allowed regions, shaded bands, and region demarcations.
- `c_esc*` colors are preferred for text labels or annotations associated with those regions.

The user currently often prefers markers without borders. Plot scripts should expose marker edge options such as `POINT_EDGE_COLOR` and `POINT_LINEWIDTH`, with `"none"` and `0.0` as acceptable defaults when appropriate.

## Plot script contract

For plotting tasks, the complete deliverable is not only a generated figure. The AI agent must leave a persistent, human-editable Python script and the generated PDF.

The plot script must be saved in the corresponding PlotSet folder:

```text
outputs/plots/<AnalysisName>/<PlotSetName>/
```

The script must be understandable and editable by the user without inspecting hidden helper scripts, temporary notebooks or agent-only generators.

The user's plotting helpers, notebooks and style files define a visual grammar. They are references for how plots should look, not mandatory APIs. By default, the AI agent should not call helper functions such as `sp_scatter`, `cb_scatter`, `del_reg`, `style_legend` or similar functions that hide the main visual construction of the plot. Use explicit Matplotlib commands instead, unless the user explicitly requests helper-based plotting.

The AI agent may use `.mplstyle` files, reuse the user's color palette, and copy visual patterns explicitly from examples. Small local helper functions are acceptable for data loading or simple validation, but not for hiding the construction of the plot.

The AI agent must not create hidden plot scripts, hidden notebooks, temporary plotting frameworks, plot recipe dispatchers, or agent-only generators. It must not generate a plot from inline Python without saving a persistent script.

The AI agent must not export PNG files and must not leave commented PNG export lines in plot scripts. Plot export should be PDF-only unless the user explicitly requests another format.

Each generated plot script should follow this order:

1. global plot configuration;
2. data loading and preparation;
3. global style;
4. figure and axes;
5. main plots;
6. lines and regions;
7. texts, annotations and benchmark markers;
8. axes and legend;
9. PDF export.

Use visible section markers:

```python
# === GLOBAL PLOT CONFIGURATION ===
# === DATA LOADING AND PREPARATION ===
# === STYLE ===
# === FIGURE AND AXES ===
# === PLOTS ===
# === LINES AND REGIONS ===
# === TEXTS, ANNOTATIONS AND MARKERS ===
# === AXES AND LEGEND ===
# === EXPORT ===
```

For one or a few requested plots, prefer one script per figure or per figure family. For systematic plot batches, a single persistent batch script is acceptable, but all plot choices must remain explicit and editable through clear constants or lists.

## Location of plot scripts

Plot scripts should not be saved in `scripts/` by default. They should be saved inside the corresponding PlotSet folder:

```text
outputs/plots/<AnalysisName>/<PlotSetName>/<plot_script_name>.py
```

Plot scripts are usually tied to a specific dataset, scan, sampler or statistics file. Keeping each script next to its plot data makes the PlotSet reproducible. A typical layout is:

```text
outputs/plots/<AnalysisName>/
`-- HighCorrelations/
    |-- sampler_10pct.dat
    |-- stats/
    |-- plot_high_correlations.py
    |-- corr_mu__mc.pdf
    `-- corr_s12q__s23q.pdf
```

Reusable plotting utilities belong in `src/plot_tools.py`. Reusable styles belong in `styles/`. Analysis-specific plotting scripts belong in `outputs/plots/<AnalysisName>/<PlotSetName>/`.

## Creating plots

When the user requests plots, the AI agent should:

1. Identify the active Analysis or the Analysis named by the user.
2. Locate the relevant data file, preferably one with a header.
3. Locate statistics or correlation files when needed.
4. Create the plot script inside the corresponding `outputs/plots/<AnalysisName>/<PlotSetName>/` folder.
5. Use `styles/` and `src/plot_tools.py` as visual references unless helper-based plotting is explicitly requested.
6. Keep all plot configuration at the top of the plot script.
7. Export figures as PDF by default.
8. Print a short summary of generated files.

For correlation plots, the AI agent should read the sampler or scan data, select variables using the requested threshold, avoid trivial pairs such as `chi` versus `cost_function` unless explicitly requested, and add experimental lines when plotted variables correspond to active targets. Experimental values must come from `src/targets.py`, not from hardcoded values.

## Outputs

The AI agent must not delete or overwrite existing outputs unless the user explicitly asks. If a plot file already exists, the AI agent should ask before overwriting it or create a clearly versioned filename.

Generated plots should normally be saved as PDF using `dpi=300` and `bbox_inches="tight"`.

## Forbidden hidden work

The agent must not create persistent artifacts whose purpose is only to make the agent's work easier while hiding the real workflow from the user.

Forbidden by default:

- hidden analyses;
- hidden scripts;
- hidden notebooks;
- hidden configuration files;
- agent-only helper modules;
- temporary persistent model registries;
- private task runners;
- hidden plotting generators;
- selectors for physics variants not requested by the user;
- wrappers that import model definitions from task scripts;
- alternative execution paths that bypass the documented architecture.

Allowed for inspection:

- shell commands;
- inline Python commands;
- temporary files outside the repository, provided they are deleted and reported if relevant.

Any persistent file created inside the repository must be placed in a documented location and reported at the end of the task.

## Required reporting

At the end of a task, the agent must report:

1. files modified;
2. files created;
3. files deleted, if any;
4. commands executed;
5. validation results;
6. outputs generated;
7. any persistent helper file created;
8. any change made outside a `USER/AGENT CONFIGURATION` block;
9. any assumption or unresolved issue requiring human review.

For model-definition tasks, the report must also state whether `src/model_definitions.py` and/or `src/likelihood.py::_build_values` were modified.

For penalty tasks, the report must include the penalty condition, mathematical form, coefficient and dictionary key.

## Reporting

After each task, the AI agent should report:

- files modified;
- scripts executed;
- input data used;
- outputs generated;
- errors or warnings;
- whether `cost_function = chi + penalty_total` was checked when relevant.

For plotting tasks, the AI agent should report:

- plot script created;
- data file used;
- variables plotted;
- PDFs generated;
- output folder.
