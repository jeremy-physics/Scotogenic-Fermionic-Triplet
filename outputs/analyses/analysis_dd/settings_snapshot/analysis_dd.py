ANALYSIS_NAME = "analysis_dd"

MODEL_VERSION = "fermion_triplet_direct_detection"
BOUNDS_VERSION = "fermion_triplet_bounds_v1"
TARGETS_VERSION = "fermion_triplet_targets_v1"

CUSTOM_BOUNDS = {}

OUTPUT_ROOT = "outputs/analyses"

ACTIVE_OBSERVABLES = [
    "neutrinos",
    "scalars",
    "lfv",
    #"direct_detection",
]

ACTIVE_CHI_TARGET_SETS = None
CHI_SIGMA_SCALE = 3.0
CUSTOM_CHI_TARGETS = {}

RAW_VARIABLE_SETS = [
    "raw_variables",
    "lepton_flavor_violation",
]
SHOW_VARIABLE_SETS = [
    "active_chi_targets",
    "scalar_masses",
    "scalar_mixing",
    #"neutrino_yukawas",
    "direct_detection",
    "lepton_flavor_violation",
    "cost_breakdown",
]
EXTRA_RAW_VARIABLES = []
EXTRA_SHOW_VARIABLES = []

# Fixed fermion mass used by the analytic direct-detection implementation.
FERMION_DM_MASS = 2600.0

SAVE_SOURCE_SNAPSHOT = True
RESUME_ANALYSIS = True
