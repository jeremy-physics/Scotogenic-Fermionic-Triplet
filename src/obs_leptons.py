import numpy as np
from numpy.linalg import LinAlgError, svd


# ---------------------------------------------------------
# Active-neutrino observables
# ---------------------------------------------------------


def LNobs(Mneu):
    trivial_res = np.zeros(9)

    try:
        u_neu, s_neu, _ = svd(Mneu)
    except (LinAlgError, ValueError, FloatingPointError):
        return trivial_res, 1e20

    if not np.all(np.isfinite(s_neu)) or np.any(s_neu < -1e-12):
        return trivial_res, 1e20

    order_neu = np.argsort(s_neu)
    a_neu = u_neu[:, order_neu]

    m1nu, m2nu, m3nu = np.sqrt(np.sort(s_neu))
    dm21 = m2nu**2 - m1nu**2
    dm31 = m3nu**2 - m1nu**2

    vpmns = a_neu
    jl = np.imag(vpmns[0, 1] * vpmns[1, 2] * np.conj(vpmns[0, 2] * vpmns[1, 1]))
    s13l = np.abs(vpmns[0, 2]) ** 2

    if 1.0 - s13l <= 0.0:
        return trivial_res, 1e20

    s12l = np.abs(vpmns[0, 1]) ** 2 / (1.0 - s13l)
    s23l = np.abs(vpmns[1, 2]) ** 2 / (1.0 - s13l)

    denominator = np.sqrt(s12l * s23l * s13l * (1.0 - s12l) * (1.0 - s23l)) * (1.0 - s13l)
    if not np.isfinite(denominator) or denominator < 1e-10:
        return trivial_res, 1e20

    sine_delta = jl / denominator
    deltaCP = np.pi - np.arcsin(np.clip(sine_delta, -1.0, 1.0))

    return [m1nu, m2nu, m3nu, dm21, dm31, s13l, s12l, s23l, deltaCP], 0.0
