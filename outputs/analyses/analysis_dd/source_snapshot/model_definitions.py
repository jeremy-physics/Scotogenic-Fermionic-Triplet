import numpy as np
from numpy.linalg import svd


# ---------------------------------------------------------
# Scalar sector
# ---------------------------------------------------------


def MVIS(v, vsigma, lambda_phi, lambda_phi_sigma, lambda_sigma):
    return np.array(
        [
            [2.0 * v**2 * lambda_phi, v * vsigma * lambda_phi_sigma],
            [v * vsigma * lambda_phi_sigma, 2.0 * vsigma**2 * lambda_sigma],
        ],
        dtype=float,
    )


def ME(
    v,
    vsigma,
    A1,
    A2,
    C,
    lambda_eta1_phi,
    lambda_eta1_sigma,
    mu_eta1,
    lambda_12_phi,
    lambda_12_sigma,
    mu_12,
    lambda_eta2_phi,
    lambda_eta2_sigma,
    mu_eta2,
    mu_phi,
):
    return np.array(
        [
            [
                0.5 * (v**2 * lambda_eta1_phi + vsigma**2 * lambda_eta1_sigma + 2.0 * mu_eta1**2),
                0.5 * (v**2 * lambda_12_phi + vsigma**2 * lambda_12_sigma + 2.0 * mu_12**2),
                (v / np.sqrt(2.0)) * A1,
            ],
            [
                0.5 * (v**2 * lambda_12_phi + vsigma**2 * lambda_12_sigma + 2.0 * mu_12**2),
                0.5 * (v**2 * lambda_eta2_phi + vsigma**2 * lambda_eta2_sigma + 2.0 * mu_eta2**2),
                (v / np.sqrt(2.0)) * A2,
            ],
            [
                (v / np.sqrt(2.0)) * A1,
                (v / np.sqrt(2.0)) * A2,
                np.sqrt(2.0) * C * vsigma + mu_phi**2,
            ],
        ],
        dtype=float,
    )


def MO(
    v,
    vsigma,
    A1,
    A2,
    C,
    lambda_eta1_phi,
    lambda_eta1_sigma,
    mu_eta1,
    lambda_12_phi,
    lambda_12_sigma,
    mu_12,
    lambda_eta2_phi,
    lambda_eta2_sigma,
    mu_eta2,
    mu_phi,
):
    return np.array(
        [
            [
                0.5 * (v**2 * lambda_eta1_phi + vsigma**2 * lambda_eta1_sigma + 2.0 * mu_eta1**2),
                0.5 * (v**2 * lambda_12_phi + vsigma**2 * lambda_12_sigma + 2.0 * mu_12**2),
                -(v / np.sqrt(2.0)) * A1,
            ],
            [
                0.5 * (v**2 * lambda_12_phi + vsigma**2 * lambda_12_sigma + 2.0 * mu_12**2),
                0.5 * (v**2 * lambda_eta2_phi + vsigma**2 * lambda_eta2_sigma + 2.0 * mu_eta2**2),
                -(v / np.sqrt(2.0)) * A2,
            ],
            [
                -(v / np.sqrt(2.0)) * A1,
                -(v / np.sqrt(2.0)) * A2,
                -np.sqrt(2.0) * C * vsigma + mu_phi**2,
            ],
        ],
        dtype=float,
    )


def MCH(
    v,
    vsigma,
    lambda_eta1_phi,
    lambda_eta1_sigma,
    lambda_12_phi,
    lambda_12_sigma,
    lambda_eta2_phi,
    lambda_eta2_sigma,
    mu_eta1,
    mu_eta2,
    mu_12,
):
    return np.array(
        [
            [
                mu_eta1**2 + 0.5 * (v**2 * lambda_eta1_phi + vsigma**2 * lambda_eta1_sigma),
                mu_12**2 + 0.5 * (v**2 * lambda_12_phi + vsigma**2 * lambda_12_sigma),
            ],
            [
                mu_12**2 + 0.5 * (v**2 * lambda_12_phi + vsigma**2 * lambda_12_sigma),
                mu_eta2**2 + 0.5 * (v**2 * lambda_eta2_phi + vsigma**2 * lambda_eta2_sigma),
            ],
        ],
        dtype=float,
    )


def acople_trilineal_h_CH_CH(
    v,
    v_sigma,
    theta_c,
    theta_1,
    lambda_12_sigma,
    lambda_12_phi,
    lambda_sigma_eta1,
    lambda_phi_eta1,
    lambda_sigma_eta2,
    lambda_phi_eta2,
):
    c_tc = np.cos(theta_c)
    s_tc = np.sin(theta_c)
    c_t1 = np.cos(theta_1)
    s_t1 = np.sin(theta_1)

    term1 = 2.0 * v_sigma * c_tc * s_t1 * s_tc * lambda_12_sigma
    term2 = 2.0 * v * c_t1 * c_tc * s_tc * lambda_12_phi
    term3 = v_sigma * c_tc**2 * s_t1 * lambda_sigma_eta1
    term4 = v * c_t1 * c_tc**2 * lambda_phi_eta1
    term5 = v_sigma * s_t1 * s_tc**2 * lambda_sigma_eta2
    term6 = v * c_t1 * s_tc**2 * lambda_phi_eta2
    return float(term1 + term2 + term3 + term4 + term5 + term6)


def acople_trilineal_h_CH2_CH2(
    v,
    v_sigma,
    theta_c,
    theta_1,
    lambda_12_sigma,
    lambda_12_phi,
    lambda_sigma_eta1,
    lambda_phi_eta1,
    lambda_sigma_eta2,
    lambda_phi_eta2,
):
    c_tc = np.cos(theta_c)
    s_tc = np.sin(theta_c)
    c_t1 = np.cos(theta_1)
    s_t1 = np.sin(theta_1)

    term1 = -2.0 * v_sigma * c_tc * s_t1 * s_tc * lambda_12_sigma
    term2 = -2.0 * v * c_t1 * c_tc * s_tc * lambda_12_phi
    term3 = v_sigma * s_t1 * s_tc**2 * lambda_sigma_eta1
    term4 = v * c_t1 * s_tc**2 * lambda_phi_eta1
    term5 = v_sigma * c_tc**2 * s_t1 * lambda_sigma_eta2
    term6 = v * c_t1 * c_tc**2 * lambda_phi_eta2
    return float(term1 + term2 + term3 + term4 + term5 + term6)


# ---------------------------------------------------------
# Neutrino sector
# ---------------------------------------------------------


def LAMBDA(mN, mI, mR, RI, RR):
    rr_sub = np.asarray(RR, dtype=complex)[:2, :]
    ri_sub = np.asarray(RI, dtype=complex)[:2, :]

    mR2 = np.asarray(mR, dtype=float) ** 2
    mI2 = np.asarray(mI, dtype=float) ** 2
    mN2 = float(mN) ** 2

    def loop_func(mk2):
        mask_degenerate = np.isclose(mk2, mN2, rtol=1e-5, atol=1e-8)
        result = np.empty_like(mk2, dtype=float)
        result[~mask_degenerate] = (
            mk2[~mask_degenerate]
            / (mk2[~mask_degenerate] - mN2)
            * np.log(mk2[~mask_degenerate] / mN2)
        )
        result[mask_degenerate] = 1.0
        return result

    fR = loop_func(mR2)
    fI = loop_func(mI2)

    term_r = rr_sub @ np.diag(fR) @ rr_sub.T
    term_i = ri_sub @ np.diag(fI) @ ri_sub.T
    return (mN / (16.0 * np.pi**2)) * (term_r - term_i)


def R_casas_ibarra(z):
    cz = np.cos(z)
    sz = np.sin(z)
    return np.array(
        [
            [0.0, cz, -sz],
            [0.0, sz, cz],
        ],
        dtype=complex,
    )


def build_pmns_matrix(s12_sq, s23_sq, s13_sq, delta_cp):
    theta12 = np.arcsin(np.sqrt(s12_sq))
    theta23 = np.arcsin(np.sqrt(s23_sq))
    theta13 = np.arcsin(np.sqrt(s13_sq))

    c12, c23, c13 = np.cos(theta12), np.cos(theta23), np.cos(theta13)
    s12, s23, s13 = np.sin(theta12), np.sin(theta23), np.sin(theta13)

    r23 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c23, s23],
            [0.0, -s23, c23],
        ],
        dtype=complex,
    )
    r13 = np.array(
        [
            [c13, 0.0, s13 * np.exp(-1j * delta_cp)],
            [0.0, 1.0, 0.0],
            [-s13 * np.exp(1j * delta_cp), 0.0, c13],
        ],
        dtype=complex,
    )
    r12 = np.array(
        [
            [c12, s12, 0.0],
            [-s12, c12, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )
    return r23 @ r13 @ r12


def build_mnu_diagonal(dm21_sq, dm31_sq):
    return np.diag([0.0, np.sqrt(dm21_sq), np.sqrt(dm31_sq)])


def compute_Yukawa_CI(U_exp, m_nu_diag, Lamb, z):
    u_lamb, s_lamb, _ = svd(Lamb)
    order = np.argsort(s_lamb)
    s_sorted = s_lamb[order]
    a_lamb = u_lamb[:, order]

    sqrt_mnu = np.sqrt(m_nu_diag)
    r_matrix = R_casas_ibarra(z)
    sqrt_inv_lamb = np.diag(np.sqrt(1.0 / (s_sorted + 1e-20)))
    r_lamb = a_lamb.T

    return np.conj(U_exp) @ sqrt_mnu @ r_matrix.T @ sqrt_inv_lamb @ r_lamb


def build_mnu_active(Y, Lamb):
    return Y @ Lamb @ Y.T
