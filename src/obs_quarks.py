import numpy as np
from numpy.linalg import svd, LinAlgError

# ---------------------------------------------------------
# Observables quarks
# ---------------------------------------------------------

def Qobs(Mu, Md):
    trivial_res = np.zeros(10)

    try:
        Uu, Su, _ = svd(Mu)
        Ud, Sd, _ = svd(Md)
    except LinAlgError:
        return trivial_res, 1e20
    
    if np.any(Su < -1e-12) or np.any(Sd < -1e-12):
        return trivial_res, 1e20
    
    idu = np.argsort(Su)
    Au = Uu[:, idu]

    idd = np.argsort(Sd)
    Ad = Ud[:, idd]

    mu, mc, mt = np.sqrt(np.sort(Su))
    md, ms, mb = np.sqrt(np.sort(Sd))

    VCKM = Au.conj().T @ Ad
    Jq = np.imag(VCKM[0,1]*VCKM[1,2]*np.conj(VCKM[0,2]*VCKM[1,1]))
    s13q = np.abs(VCKM[0,2])
    s12q = np.sqrt(np.abs(VCKM[0,1])**2 / (1 - s13q**2))
    s23q = np.sqrt(np.abs(VCKM[1,2])**2 / (1 - s13q**2))

    return [mu, mc, mt, md, ms, mb, s13q, s12q, s23q, Jq], 0.0