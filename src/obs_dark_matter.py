import numpy as np


# ---------------------------------------------------------
# Parameters and running functions
# ---------------------------------------------------------

pi = np.pi
zeta3 = 1.2020569031595942854
zeta4 = pi**4 / 90.0
zeta5 = 1.0369277551433699263


def beta0(nf):
    return 11.0 - (2.0 / 3.0) * nf


def beta1(nf):
    return 102.0 - (38.0 / 3.0) * nf


def beta2(nf):
    return 2857.0 / 2.0 - (5033.0 / 18.0) * nf + (325.0 / 54.0) * nf**2


def beta3(nf):
    return (
        149753.0 / 6.0
        + 3564.0 * zeta3
        - (1078361.0 / 162.0 + 6508.0 / 27.0 * zeta3) * nf
        + (50065.0 / 162.0 + 6472.0 / 81.0 * zeta3) * nf**2
        + 1093.0 / 729.0 * nf**3
    )


d2 = 11.0 / 72.0


def d3(nf):
    return (
        564731.0 / 124416.0
        - 82043.0 / 27648.0 * zeta3
        - 2633.0 / 31104.0 * nf
    )


def gamma0(nf):
    return 8.0


def gamma1(nf):
    return 404.0 / 3.0 - (40.0 / 9.0) * nf


def gamma2(nf):
    return 2498.0 - (4432.0 / 27.0 + 320.0 / 3.0 * zeta3) * nf - (
        280.0 / 81.0
    ) * nf**2


def gamma3(nf):
    return (
        4603055.0 / 81.0
        + 271360.0 / 27.0 * zeta3
        - 17600.0 * zeta5
        + (
            -183446.0 / 27.0
            - 68384.0 / 9.0 * zeta3
            + 1760.0 * zeta4
            + 36800.0 / 9.0 * zeta5
        )
        * nf
        + (10484.0 / 243.0 + 1600.0 / 9.0 * zeta3 - 320.0 / 3.0 * zeta4)
        * nf**2
        + (-664.0 / 243.0 + 128.0 / 27.0 * zeta3) * nf**3
    )


MqQ2 = 5.0 / 18.0


def MqQ3(n):
    return 311.0 / 1296.0 + 5.0 / 3.0 * zeta3 + 53.0 / 216.0 * n


MGQ1 = 11.0 / 4.0


def MGQ2(n):
    return 2777.0 / 288.0 - 67.0 / 96.0 * n


def MGQ3(n):
    return (
        897943.0 / 9216.0 * zeta3
        - 2761331.0 / 41472.0
        + n * (58723.0 / 20736.0 - 110779.0 / 13824.0 * zeta3)
        - n**2 * 6865.0 / 31104.0
    )


def FR(M, x):
    return drop_matrix(M, -(x + 1), -2, -(x + 1), -2)


def FM(M, x):
    return drop_matrix(M, -(x + 1), -2, -(x + 2), -3)


def beta_t(n, alpha_s):
    return (
        -beta0(n) * alpha_s / (4.0 * pi)
        - beta1(n) * (alpha_s / (4.0 * pi)) ** 2
        - beta2(n) * (alpha_s / (4.0 * pi)) ** 3
        - beta3(n) * (alpha_s / (4.0 * pi)) ** 4
    )


def gamma(n, alpha_s):
    return (
        -gamma0(n) * alpha_s / (4.0 * pi)
        - gamma1(n) * (alpha_s / (4.0 * pi)) ** 2
        - gamma2(n) * (alpha_s / (4.0 * pi)) ** 3
        - gamma3(n) * (alpha_s / (4.0 * pi)) ** 4
    )


def Rqg(n, alpha_h, alpha_l):
    return (
        2.0
        * (gamma(n, alpha_h) - gamma(n, alpha_l))
        * alpha_h
        / (beta_t(n, alpha_h) * pi)
    )


def Rgg(n, alpha_h, alpha_l):
    return beta_t(n, alpha_l) / beta_t(n, alpha_h) * alpha_h / alpha_l


def MqQ(alpha_s, n):
    return (alpha_s / pi) ** 2 * MqQ2 + (alpha_s / pi) ** 3 * MqQ3(n)


def MGQ(n, alpha_h, alpha_l):
    return (
        -(1.0 / 12.0)
        * alpha_h
        / alpha_l
        * (
            1.0
            + MGQ1 * (alpha_h / pi)
            + MGQ2(n) * (alpha_h / pi) ** 2
            + MGQ3(n) * (alpha_h / pi) ** 3
        )
    )


def MqG(n, alpha_l, alpha_h):
    return (
        2.0
        * alpha_h
        / (pi * beta_t(n + 1, alpha_h))
        * (
            (1.0 - gamma(n, alpha_l))
            - (1.0 - gamma(n + 1, alpha_h)) * (1.0 + MqQ(alpha_h, n))
        )
    )


def MGG(n, alpha_l, alpha_h):
    return (
        alpha_h
        / alpha_l
        * beta_t(n, alpha_l)
        / beta_t(n + 1, alpha_h)
        - 2.0
        * alpha_h
        / (pi * beta_t(n + 1, alpha_h))
        * (1.0 - gamma(n + 1, alpha_h))
        * MGQ(n, alpha_h, alpha_l)
    )


def drop_matrix(M, row_start, row_end, col_start, col_end):
    M = np.asarray(M, dtype=float)
    row_drop = range(M.shape[0] + row_start, M.shape[0] + row_end + 1)
    col_drop = range(M.shape[1] + col_start, M.shape[1] + col_end + 1)
    rows = np.delete(np.arange(M.shape[0]), list(row_drop))
    cols = np.delete(np.arange(M.shape[1]), list(col_drop))
    return M[np.ix_(rows, cols)]


def Rq(n, alpha_h, alpha_l):
    M = np.zeros((6, 6), dtype=float)
    rqg = Rqg(n, alpha_h, alpha_l)

    for i in range(5):
        M[i, i] = 1.0
        M[i, 5] = rqg

    M[5, 5] = Rgg(n, alpha_h, alpha_l)
    return M


def Mq(n, alpha_l, alpha_h):
    M = np.zeros((6, 7), dtype=float)
    mqq = MqQ(alpha_h, n)
    mqg = MqG(n, alpha_l, alpha_h)

    for i in range(5):
        M[i, i] = 1.0
        M[i, 5] = mqq
        M[i, 6] = mqg

    M[5, 5] = MGQ(n, alpha_h, alpha_l)
    M[5, 6] = MGG(n, alpha_l, alpha_h)
    return M


# ---------------------------------------------------------
# Model parameters
# ---------------------------------------------------------

alphaEM = 1.0 / 128.0
mN = 0.939
v = 246.22
mW = 80.3692
mZ = 91.1876
mb = 4.183
mc = 1.273
mt = 172.56
muhad = 1.0
sw = np.sqrt(0.23122)
mXi = 2600.0
mh = 125.2
Cfact = 0.389379e-27
alpha2 = mW**2 / (pi * v**2)


# Strong running coupling: fixed final values from In[244], Out[244].
alpha5mz = 0.118
alpha5mb = 0.224569
alpha4mb = 0.224796
alpha4mc = 0.386944
alpha3mc = 0.38836
alpha3muhad = 0.476713


def beta_alpha(nf, alpha_s):
    return (
        -(beta0(nf) / (2.0 * pi)) * alpha_s**2
        - beta1(nf) / (8.0 * pi**2) * alpha_s**3
        - beta2(nf) / (32.0 * pi**3) * alpha_s**4
        - beta3(nf) / (128.0 * pi**4) * alpha_s**5
    )


def gamma_alpha(n, alpha_s):
    return gamma(n, alpha_s)


# ---------------------------------------------------------
# One-loop contribution
# ---------------------------------------------------------

fTq = 0.0423 + 0.0563


def fq(alpha, alpha_s, mh_light, mH):
    return (
        -((pi * alpha2**2) / (2.0 * mW))
        * (1.0 + 0.39 * alpha_s)
        * (np.cos(alpha) ** 2 / mh_light**2 + np.sin(alpha) ** 2 / mH**2)
    )


def fb(alpha, alpha_s, mh_light, mH):
    return (
        -((pi * alpha2**2) / (2.0 * mW))
        * (1.0 + 0.003 * alpha_s)
        * (np.cos(alpha) ** 2 / mh_light**2 + np.sin(alpha) ** 2 / mH**2)
    )


def fG(alpha, alpha_s, mh_light, mH):
    return (
        alpha2**2
        / (4.0 * mW)
        * (1.86 + 0.93 * alpha_s)
        * (np.cos(alpha) ** 2 / mh_light**2 + np.sin(alpha) ** 2 / mH**2)
    )


Q = np.array([0.223, 0.118, 0.0258, 0.0187, 0.0117], dtype=float)
Qb = np.array([0.036, 0.037, 0.0258, 0.0187, 0.0117], dtype=float)


def g1(alpha_s):
    return np.array(
        [
            alpha2**2 / mW**3 * (1.05 + 0.85 * alpha_s),
            alpha2**2 / mW**3 * (1.05 + 0.85 * alpha_s),
            alpha2**2 / mW**3 * (1.05 + 0.85 * alpha_s),
            alpha2**2 / mW**3 * (1.05 + 0.85 * alpha_s),
            alpha2**2 / mW**3 * (0.14 - 0.005 * alpha_s),
        ],
        dtype=float,
    )


def g2(alpha_s):
    return np.array(
        [
            (2.0 * alpha2**2) / (27.0 * mW**3) * alpha_s,
            (2.0 * alpha2**2) / (27.0 * mW**3) * alpha_s,
            (2.0 * alpha2**2) / (27.0 * mW**3) * alpha_s,
            (2.0 * alpha2**2) / (27.0 * mW**3) * alpha_s,
            0.0,
        ],
        dtype=float,
    )


def g1G(alpha_s):
    return alpha2**2 / (4.0 * mW**3) * 2.66 * alpha_s


def g2G(alpha_s):
    return -(alpha2**2 / (9.0 * mW**3)) * alpha_s


g2mz = 0.464


def fTG(nf, alpha_s):
    return (
        (4.0 * alpha_s**2)
        / (pi * beta_alpha(nf, alpha_s))
        * (1.0 - (1.0 - gamma_alpha(nf, alpha_s)) * fTq)
    )


def C0_wolfram(mXi_value, alpha, vsigma, mh_light, mh2):
    return (
        -(mXi_value / (4.0 * v * vsigma))
        * np.sin(2.0 * alpha)
        * (1.0 / mh_light**2 - 1.0 / mh2**2)
    )


def ew_coefficients(alpha, mh_light, mH):
    Cqz5 = np.array(
        [
            [fq(alpha, alpha5mz, mh_light, mH)],
            [fq(alpha, alpha5mz, mh_light, mH)],
            [fq(alpha, alpha5mz, mh_light, mH)],
            [fq(alpha, alpha5mz, mh_light, mH)],
            [fb(alpha, alpha5mz, mh_light, mH)],
            [fG(alpha, alpha5mz, mh_light, mH)],
        ],
        dtype=float,
    )
    Cqb4 = FM(Mq(4, alpha4mb, alpha5mb), 1) @ Rq(5, alpha5mz, alpha5mb) @ Cqz5
    Cqc3 = (
        FM(Mq(3, alpha3mc, alpha4mc), 2)
        @ FR(Rq(4, alpha4mb, alpha4mc), 1)
        @ Cqb4
    )
    Cqhad3 = FR(Rq(3, alpha3mc, alpha3muhad), 2) @ Cqc3
    return Cqhad3[-2, 0], Cqhad3[-1, 0]


def tree_coefficients(mXi_value, alpha, vsigma, mh_light, mh2):
    c0 = C0_wolfram(mXi_value, alpha, vsigma, mh_light, mh2)
    Cqtreet5 = -np.array(
        [[c0], [c0], [c0], [c0], [c0], [-(1.0 / 8.0) * c0]],
        dtype=float,
    )
    Cqtreeb4 = (
        FM(Mq(4, alpha4mb, alpha5mb), 1) @ Rq(5, alpha5mz, alpha5mb) @ Cqtreet5
    )
    Cqtreec3 = (
        FM(Mq(3, alpha3mc, alpha4mc), 2)
        @ FR(Rq(4, alpha4mb, alpha4mc), 1)
        @ Cqtreeb4
    )
    Cqtreehad3 = FR(Rq(3, alpha3mc, alpha3muhad), 2) @ Cqtreec3
    return Cqtreehad3[-2, 0], Cqtreehad3[-1, 0]


def FL(mXi_value, x):
    if not np.isfinite(mXi_value) or not np.isfinite(x) or x <= 0.0:
        return np.nan

    r = (mXi_value / x) ** 2
    if not np.isfinite(r):
        return np.nan
    if r == 0.0:
        return -0.5 / x**2
    if r > 1.0:
        return np.nan
    if r == 1.0:
        return -1.0 / x**2

    one_minus_r = 1.0 - r
    return -(1.0 / x**2) * (r + one_minus_r * np.log(one_minus_r)) / (r**2)


def fqscot(RR, RI, yXi, MR, MI, alpha, gRh, gRH, gIh, gIH, mh_light, mH, mXi_value):
    RR = np.asarray(RR, dtype=complex)
    RI = np.asarray(RI, dtype=complex)
    yXi = np.asarray(yXi, dtype=complex)
    MR = np.asarray(MR, dtype=float)
    MI = np.asarray(MI, dtype=float)
    gRh = np.asarray(gRh, dtype=float)
    gRH = np.asarray(gRH, dtype=float)
    gIh = np.asarray(gIh, dtype=float)
    gIH = np.asarray(gIH, dtype=float)

    total = 0.0 + 0j
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)

    for l in range(3):
        for a in range(3):
            yR = sum(RR[n, a] * yXi[l, n] for n in range(2))
            yI = sum(RI[n, a] * yXi[l, n] for n in range(2))
            total += (
                -np.abs(yR) ** 2
                * FL(mXi_value, MR[a])
                * (gRh[a] * c_alpha / mh_light**2 - gRH[a] * s_alpha / mH**2)
                + np.abs(yI) ** 2
                * FL(mXi_value, MI[a])
                * (gIh[a] * c_alpha / mh_light**2 - gIH[a] * s_alpha / mH**2)
            )

    return mXi_value / (64.0 * pi**2 * v) * total


def f_scoto(
    mXi_value,
    alpha,
    mh_light,
    mH,
    RR,
    RI,
    yXi,
    MR,
    MI,
    A1,
    A2,
    C,
    lamb_eta1_phi,
    lamb_eta1_sigma,
    lamb_12_phi,
    lamb_12_sigma,
    lamb_eta2_phi,
    lamb_eta2_sigma,
    vsigma,
):
    RR = np.asarray(RR, dtype=float)
    RI = np.asarray(RI, dtype=float)
    gRh = np.zeros(3, dtype=float)
    gRH = np.zeros(3, dtype=float)
    gIh = np.zeros(3, dtype=float)
    gIH = np.zeros(3, dtype=float)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)

    for a in range(3):
        r1, r2, r3 = RR[0, a], RR[1, a], RR[2, a]
        phi_R = (
            np.sqrt(2.0) * A1 * r1 * r3
            + np.sqrt(2.0) * A2 * r2 * r3
            + 2.0 * r1 * r2 * v * lamb_12_phi
            + r1**2 * v * lamb_eta1_phi
            + r2**2 * v * lamb_eta2_phi
        )
        sigma_R = (
            2.0 * C * r3**2
            + vsigma
            * (
                2.0 * r1 * r2 * lamb_12_sigma
                + r1**2 * lamb_eta1_sigma
                + r2**2 * lamb_eta2_sigma
            )
        )
        gRh[a] = phi_R * c_alpha + sigma_R * s_alpha
        gRH[a] = sigma_R * c_alpha - phi_R * s_alpha

        i1, i2, i3 = RI[0, a], RI[1, a], RI[2, a]
        phi_I = (
            -np.sqrt(2.0) * A1 * i1 * i3
            - np.sqrt(2.0) * A2 * i2 * i3
            + 2.0 * i1 * i2 * v * lamb_12_phi
            + i1**2 * v * lamb_eta1_phi
            + i2**2 * v * lamb_eta2_phi
        )
        sigma_I = (
            -2.0 * C * i3**2
            + vsigma
            * (
                2.0 * i1 * i2 * lamb_12_sigma
                + i1**2 * lamb_eta1_sigma
                + i2**2 * lamb_eta2_sigma
            )
        )
        gIh[a] = phi_I * c_alpha + sigma_I * s_alpha
        gIH[a] = sigma_I * c_alpha - phi_I * s_alpha

    fq_scoto = fqscot(
        RR,
        RI,
        yXi,
        MR,
        MI,
        alpha,
        gRh,
        gRH,
        gIh,
        gIH,
        mh_light,
        mH,
        mXi_value,
    )
    Cqscoto5 = -np.array(
        [
            [fq_scoto],
            [fq_scoto],
            [fq_scoto],
            [fq_scoto],
            [fq_scoto],
            [-(1.0 / 8.0) * fq_scoto],
        ],
        dtype=complex,
    )
    Cqscotob4 = (
        FM(Mq(4, alpha4mb, alpha5mb), 1) @ Rq(5, alpha5mz, alpha5mb) @ Cqscoto5
    )
    Cqscotoc3 = (
        FM(Mq(3, alpha3mc, alpha4mc), 2)
        @ FR(Rq(4, alpha4mb, alpha4mc), 1)
        @ Cqscotob4
    )
    Cqscotohad3 = FR(Rq(3, alpha3mc, alpha3muhad), 2) @ Cqscotoc3
    CscotoQ = Cqscotohad3[-2, 0]
    CscotoG = Cqscotohad3[-1, 0]
    kSscot = CscotoQ * fTq + CscotoG * fTG(3, alpha3muhad)
    return float(np.real(mN * kSscot))


def f_EW(alpha, mh_light, mH):
    fqR, fGR = ew_coefficients(alpha, mh_light, mH)
    kS = fqR * fTq + fGR * fTG(3, alpha3muhad)
    kT = (3.0 / 4.0) * np.sum((Q + Qb) * (g1(alpha5mz) + g2(alpha5mz))) - (
        3.0 / 4.0
    ) * g2mz * (g1G(alpha5mz) + g2G(alpha5mz))
    return mN * (kS + kT)


def f_tree(mXi_value, vsigma, alpha, mh_light, mH):
    C0Q, C0G = tree_coefficients(mXi_value, alpha, vsigma, mh_light, mH)
    kSs = C0Q * fTq + C0G * fTG(3, alpha3muhad)
    return mN * kSs


def sigma_SIf(
    mXi_value,
    vsigma,
    alpha,
    mh_light,
    mH,
    RR,
    RI,
    yXi,
    MR,
    MI,
    A1,
    A2,
    C,
    lamb_eta1_phi,
    lamb_eta1_sigma,
    lamb_12_phi,
    lamb_12_sigma,
    lamb_eta2_phi,
    lamb_eta2_sigma,
):
    fEW = f_EW(alpha, mh_light, mH)
    ftree = f_tree(mXi_value, vsigma, alpha, mh_light, mH)
    fscoto = f_scoto(
        mXi_value,
        alpha,
        mh_light,
        mH,
        RR,
        RI,
        yXi,
        MR,
        MI,
        A1,
        A2,
        C,
        lamb_eta1_phi,
        lamb_eta1_sigma,
        lamb_12_phi,
        lamb_12_sigma,
        lamb_eta2_phi,
        lamb_eta2_sigma,
        vsigma,
    )
    fN_TOT = fEW + ftree + fscoto
    sigma_si = 4.0 / pi * ((mXi_value * mN) / (mXi_value + mN)) ** 2 * fN_TOT**2 * Cfact
    return sigma_si, fEW, ftree, fscoto, fN_TOT


def direct_detection_explicit(
    mXi_value,
    vsigma,
    alpha,
    mh_light,
    mH,
    RR,
    RI,
    yXi,
    MR,
    MI,
    A1,
    A2,
    C,
    lamb_eta1_phi,
    lamb_eta1_sigma,
    lamb_12_phi,
    lamb_12_sigma,
    lamb_eta2_phi,
    lamb_eta2_sigma,
):
    sigma_si, fEW, ftree, fscoto, fN_TOT = sigma_SIf(
        mXi_value,
        vsigma,
        alpha,
        mh_light,
        mH,
        RR,
        RI,
        yXi,
        MR,
        MI,
        A1,
        A2,
        C,
        lamb_eta1_phi,
        lamb_eta1_sigma,
        lamb_12_phi,
        lamb_12_sigma,
        lamb_eta2_phi,
        lamb_eta2_sigma,
    )
    return fEW, ftree, fscoto, fN_TOT, sigma_si


def compute_direct_detection_observables(
    mXi_value,
    vsigma,
    alpha,
    mh_light,
    mH,
    RR,
    RI,
    yXi,
    MR,
    MI,
    A1,
    A2,
    C,
    lamb_eta1_phi,
    lamb_eta1_sigma,
    lamb_12_phi,
    lamb_12_sigma,
    lamb_eta2_phi,
    lamb_eta2_sigma,
):
    fEW, ftree, fscoto, fN_TOT, sigma_si = direct_detection_explicit(
        mXi_value,
        vsigma,
        alpha,
        mh_light,
        mH,
        RR,
        RI,
        yXi,
        MR,
        MI,
        A1,
        A2,
        C,
        lamb_eta1_phi,
        lamb_eta1_sigma,
        lamb_12_phi,
        lamb_12_sigma,
        lamb_eta2_phi,
        lamb_eta2_sigma,
    )
    sigma_si = float(sigma_si)
    return {
        "sigma_si": sigma_si,
        "sigma_SI": sigma_si,
        "direct_detection": sigma_si,
        "fEW": float(fEW),
        "ftree": float(ftree),
        "fscoto": float(fscoto),
        "fN_TOT": float(fN_TOT),
    }
