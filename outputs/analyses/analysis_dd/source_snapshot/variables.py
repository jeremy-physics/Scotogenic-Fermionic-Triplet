import numpy as np


TRILINEAR_COUPLINGS = ["A1", "A2", "C"]
COMPUTED_QUARTIC_COUPLINGS = ["lambda_phi"]
INPUT_QUARTIC_COUPLINGS = [
    "lambda_eta1_phi",
    "lambda_eta1_sigma",
    "lambda_12_phi",
    "lambda_12_sigma",
    "lambda_eta2_phi",
    "lambda_eta2_sigma",
    "lambda_phi_sigma",
    "lambda_sigma",
]
QUARTIC_COUPLINGS = COMPUTED_QUARTIC_COUPLINGS + INPUT_QUARTIC_COUPLINGS
QUADRATIC_PARAMETERS = ["mu_12", "mu_eta1", "mu_eta2", "mu_phi"]
CASAS_IBARRA_PARAMETERS = ["z"]
TRIPLET_PARAMETERS = ["zXi", "mXi", "vsigma"]

INPUT_PARAMETERS = (
    TRILINEAR_COUPLINGS
    + INPUT_QUARTIC_COUPLINGS
    + QUADRATIC_PARAMETERS
    + CASAS_IBARRA_PARAMETERS
    + ["zXi"]
)

COMPUTED_SCALAR_COUPLINGS = COMPUTED_QUARTIC_COUPLINGS
SCALAR_COUPLINGS = QUADRATIC_PARAMETERS + TRILINEAR_COUPLINGS + QUARTIC_COUPLINGS
VISIBLE_MASSES = ["mh", "mHH"]
CP_EVEN_INERT_MASSES = ["mR1", "mR2", "mR3"]
CP_ODD_INERT_MASSES = ["mI1", "mI2", "mI3"]
CHARGED_INERT_MASSES = ["mCH1", "mCH2"]
INERT_MASSES = CP_EVEN_INERT_MASSES + CP_ODD_INERT_MASSES + CHARGED_INERT_MASSES
SCALAR_MASSES = VISIBLE_MASSES + INERT_MASSES
SCALAR_MIXING = ["alpha", "theta_c", "vsigma", "mXi"]
HIGGS_OBSERVABLES = ["diph", "Rgamma", "kappaF"]

NEUTRINO_DATA = ["dm21", "dm31", "s13l", "s12l", "s23l", "deltaCP"]
NEUTRINO_MASSES = ["m1nu", "m2nu", "m3nu", "sum_nu"]
NEUTRINO_YUKAWAS = [
    "yXi11_abs",
    "yXi12_abs",
    "yXi21_abs",
    "yXi22_abs",
    "yXi31_abs",
    "yXi32_abs",
]
LAMBDA_MATRIX = ["Lambda11", "Lambda12", "Lambda21", "Lambda22"]

DIRECT_DETECTION = ["sigma_si", "fEW", "ftree", "fscoto", "fN_TOT"]
DARK_MATTER = DIRECT_DETECTION
LEPTON_FLAVOR_VIOLATION = ["BR_mu_e", "BR_tau_mu", "BR_mu_3e", "CR_muAL_e"]

COST_BREAKDOWN = [
    "chi",
    "pen_structure",
    "pen_restriction",
    "penalty_total",
    "cost_function",
]

ALL_MODEL_VARIABLES = (
    INPUT_PARAMETERS
    + SCALAR_COUPLINGS
    + SCALAR_MASSES
    + SCALAR_MIXING
    + HIGGS_OBSERVABLES
    + NEUTRINO_DATA
    + NEUTRINO_MASSES
    + NEUTRINO_YUKAWAS
    + LAMBDA_MATRIX
    + DIRECT_DETECTION
    + LEPTON_FLAVOR_VIOLATION
    + COST_BREAKDOWN
)

RAW_VARIABLES = (
    INPUT_PARAMETERS
    + SCALAR_COUPLINGS
    + SCALAR_MASSES
    + SCALAR_MIXING
    + HIGGS_OBSERVABLES
    + NEUTRINO_DATA
    + NEUTRINO_MASSES
    + NEUTRINO_YUKAWAS
    + LAMBDA_MATRIX
    + TRIPLET_PARAMETERS
    + DIRECT_DETECTION
    + LEPTON_FLAVOR_VIOLATION
    + ["cost_function"]
)

VARIABLE_LABELS = {
    "A1": r"$A_1$",
    "A2": r"$A_2$",
    "C": r"$C$",
    "lambda_phi": r"$\lambda_\phi$",
    "lambda_eta1_phi": r"$\lambda_{\eta_1\phi}$",
    "lambda_eta1_sigma": r"$\lambda_{\eta_1\sigma}$",
    "lambda_12_phi": r"$\lambda_{12\phi}$",
    "lambda_12_sigma": r"$\lambda_{12\sigma}$",
    "lambda_eta2_phi": r"$\lambda_{\eta_2\phi}$",
    "lambda_eta2_sigma": r"$\lambda_{\eta_2\sigma}$",
    "lambda_phi_sigma": r"$\lambda_{\phi\sigma}$",
    "lambda_sigma": r"$\lambda_\sigma$",
    "mu_12": r"$\mu_{12}$",
    "mu_eta1": r"$\mu_{\eta_1}$",
    "mu_eta2": r"$\mu_{\eta_2}$",
    "mu_phi": r"$\mu_\varphi$",
    "zXi": r"$z_\Xi$",
    "z": r"$z$",
    "mh": r"$m_h$",
    "mHH": r"$m_H$",
    "mR1": r"$m_{R_1}$",
    "mR2": r"$m_{R_2}$",
    "mR3": r"$m_{R_3}$",
    "mI1": r"$m_{I_1}$",
    "mI2": r"$m_{I_2}$",
    "mI3": r"$m_{I_3}$",
    "mCH1": r"$m_{H_1^\pm}$",
    "mCH2": r"$m_{H_2^\pm}$",
    "alpha": r"$\alpha$",
    "theta_c": r"$\theta_c$",
    "vsigma": r"$v_\sigma$",
    "mXi": r"$m_\Xi$",
    "diph": r"$R_{\gamma\gamma}$",
    "Rgamma": r"$R_{\gamma\gamma}$",
    "kappaF": r"$\kappa_F$",
    "dm21": r"$\Delta m_{21}^2$",
    "dm31": r"$\Delta m_{31}^2$",
    "s12l": r"$s_{12}^{\ell\,2}$",
    "s13l": r"$s_{13}^{\ell\,2}$",
    "s23l": r"$s_{23}^{\ell\,2}$",
    "deltaCP": r"$\delta_{\rm CP}$",
    "m1nu": r"$m_1$",
    "m2nu": r"$m_2$",
    "m3nu": r"$m_3$",
    "sum_nu": r"$\sum m_\nu$",
    "direct_detection": r"$\sigma_{\rm SI}$",
    "sigma_si": r"$\sigma_{\rm SI}$",
    "sigma_SI": r"$\sigma_{\rm SI}$",
    "fEW": r"$f_{\rm EW}$",
    "ftree": r"$f_{\rm tree}$",
    "fscoto": r"$f_{\rm scoto}$",
    "fN_TOT": r"$f_N^{\rm tot}$",
    "BR_mu_e": r"${\rm BR}(\mu\to e\gamma)$",
    "BR_tau_mu": r"${\rm BR}(\tau\to\mu\gamma)$",
    "BR_mu_3e": r"${\rm BR}(\mu\to 3e)$",
    "CR_muAL_e": r"${\rm CR}(\mu{\rm Al}\to e{\rm Al})$",
    "chi": r"$\chi^2$",
    "cost_function": r"$\mathcal{C}$",
    "pen_structure": r"$P_{\rm str}$",
    "pen_restriction": r"$P_{\rm res}$",
    "penalty_total": r"$P_{\rm tot}$",
}

for row in range(1, 4):
    for col in range(1, 3):
        VARIABLE_LABELS[f"yXi{row}{col}_abs"] = rf"$|Y_{{\Xi,{row}{col}}}|$"

for row in range(1, 3):
    for col in range(1, 3):
        VARIABLE_LABELS[f"Lambda{row}{col}"] = rf"$\Lambda_{{{row}{col}}}$"


def get_variable_label(name):
    return VARIABLE_LABELS.get(name, name)


def get_variable_labels(names):
    return [get_variable_label(name) for name in names]


VARIABLE_SETS = {
    "trilinear_couplings": TRILINEAR_COUPLINGS,
    "computed_quartic_couplings": COMPUTED_QUARTIC_COUPLINGS,
    "input_quartic_couplings": INPUT_QUARTIC_COUPLINGS,
    "quartic_couplings": QUARTIC_COUPLINGS,
    "quadratic_parameters": QUADRATIC_PARAMETERS,
    "casas_ibarra_parameters": CASAS_IBARRA_PARAMETERS,
    "input_parameters": INPUT_PARAMETERS,
    "scalar_couplings": SCALAR_COUPLINGS,
    "visible_masses": VISIBLE_MASSES,
    "inert_masses": INERT_MASSES,
    "cp_even_inert_masses": CP_EVEN_INERT_MASSES,
    "cp_odd_inert_masses": CP_ODD_INERT_MASSES,
    "charged_inert_masses": CHARGED_INERT_MASSES,
    "scalar_masses": SCALAR_MASSES,
    "scalar_mixing": SCALAR_MIXING,
    "higgs_observables": HIGGS_OBSERVABLES,
    "neutrino_data": NEUTRINO_DATA,
    "neutrino_masses": NEUTRINO_MASSES,
    "neutrino_yukawas": NEUTRINO_YUKAWAS,
    "lambda_matrix": LAMBDA_MATRIX,
    "direct_detection": DIRECT_DETECTION,
    "dark_matter": DARK_MATTER,
    "lepton_flavor_violation": LEPTON_FLAVOR_VIOLATION,
    "lfv": LEPTON_FLAVOR_VIOLATION,
    "raw_variables": RAW_VARIABLES,
    "cost_breakdown": COST_BREAKDOWN,
    "all_model_variables": ALL_MODEL_VARIABLES,
}

DYNAMIC_VARIABLE_SETS = {"active_chi_targets"}

DEFAULT_RAW_VARIABLE_SETS = ["active_chi_targets", "cost_breakdown"]
DEFAULT_SHOW_VARIABLE_SETS = ["active_chi_targets", "cost_breakdown"]


_WARNED_MISSING_VARIABLES = set()


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def get_active_chi_target_variables(settings=None):
    from targets import get_active_chi_model_keys

    return get_active_chi_model_keys(settings)


def get_variables_from_sets(variable_sets, extra_variables=None, settings=None):
    variables = []
    for set_name in variable_sets or []:
        if set_name == "active_chi_targets":
            variables.extend(get_active_chi_target_variables(settings))
            continue
        if set_name not in VARIABLE_SETS:
            print(f"[WARN] Variable set not found: {set_name}")
            continue
        variables.extend(VARIABLE_SETS[set_name])

    variables.extend(extra_variables or [])
    return _dedupe_preserve_order(variables)


def get_raw_variables(settings=None):
    variable_sets = getattr(settings, "RAW_VARIABLE_SETS", DEFAULT_RAW_VARIABLE_SETS)
    extra_variables = getattr(settings, "EXTRA_RAW_VARIABLES", [])
    return get_variables_from_sets(variable_sets, extra_variables, settings=settings)


def get_show_variables(settings=None):
    variable_sets = getattr(settings, "SHOW_VARIABLE_SETS", DEFAULT_SHOW_VARIABLE_SETS)
    extra_variables = getattr(settings, "EXTRA_SHOW_VARIABLES", [])
    return get_variables_from_sets(variable_sets, extra_variables, settings=settings)


def values_to_row(
    values,
    variable_names,
    missing_value=np.nan,
    requested_by="RAW_VARIABLE_SETS",
):
    row = []
    for name in variable_names:
        if name not in values:
            warning_key = (requested_by, name)
            if warning_key not in _WARNED_MISSING_VARIABLES:
                print(
                    f"[warning] variable '{name}' requested in {requested_by} "
                    "but not found in values."
                )
                _WARNED_MISSING_VARIABLES.add(warning_key)
            row.append(missing_value)
            continue

        value = values[name]
        if isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value, dtype=float).ravel()
            if array.size != 1:
                print(f"[WARN] Requested variable is not scalar: {name}")
                row.append(missing_value)
                continue
            value = array[0]

        row.append(float(value))
    return np.asarray(row, dtype=float)


def validate_row_width(variable_names, row, script_name):
    keys = list(variable_names)
    n_values = np.asarray(row).ravel().size
    if len(keys) != n_values:
        raise ValueError(
            f"[ERROR] {script_name}: header has {len(keys)} columns but "
            f"the row has {n_values} values. Requested keys: {keys}"
        )
