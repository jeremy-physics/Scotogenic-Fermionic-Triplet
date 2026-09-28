import numpy as np

# ----------------- Datos experimentales -----------------
#Quarks masses at the Z-boson scale from 2009.04851
'''
mu_exp = 1.23e-3
mc_exp = 0.620
mt_exp = 168.26
md_exp = 2.67e-3
ms_exp = 53.16e-3
mb_exp = 2.839

mu_err = 0.21e-3
mc_err = 0.017
mt_err = 0.75
md_err = 0.19e-3
ms_err = 4.61e-3
mb_err = 0.026
'''
#Quarks masses at the pole-mass scale from PDG 06-2026
mu_exp = 2.16e-3
mc_exp = 1.2729
mt_exp = 172.60
md_exp = 4.70e-3
ms_exp = 92.9e-3
mb_exp = 4.186

mu_err = 0.07e-3
mc_err = 0.0045
mt_err = 0.27
md_err = 0.07e-3
ms_err = 0.07e-3
mb_err = 0.006

#---------------------------------------

#CKM parameters from PDG 06-2026
s12q_exp = 0.22517
s23q_exp = 0.04189
s13q_exp = 0.003763

s12q_err = 0.00068
s23q_err = 0.00069
s13q_err = 0.000083

Jq_exp = 3.16e-5
Jq_err = 0.11e-5

#---------------------------------------

#Charged Lepton masses at the Z-boson scale from 2009.04851
me_exp = 0.48307e-3
mmu_exp = 0.101766
mtau_exp = 1.72856

me_err = 0.00045e-3 
mmu_err = 0.000023
mtau_err = 0.00028 

#Charged Lepton masses at the pole-mass scale from PDG 06-2026
'''
me_exp = 0.51099895069e-3
mmu_exp = 105.6583755
mtau_exp = 1776.93e-3

me_err = 0.00000000016e-3 
mmu_err = 0.0000023
mtau_err = 0.09e-3 
'''

#---------------------------------------

#Neutrino oscillation data from 2020 Global-fit 2006.11237
'''
##Normal ordering
dm21_exp = 7.50e-23
dm31_exp = 2.55e-21

dm21_err = 0.20e-23
dm31_err = 0.02e-21

s12l_exp = 0.318
s23l_exp = 0.574
s13l_exp = 0.02200

s12l_err = 0.016
s23l_err = 0.014
s13l_err = 0.00062

deltaCPl_exp = 1.08*np.pi #at 3sigma
deltaCPl_err = 0.12*np.pi

##Inverted ordering
dm21_exp = 7.50e-23
dm31_exp = 2.45e-21

dm21_err = 0.20e-23
dm31_err = 0.02e-21

s12l_exp = 0.318
s23l_exp = 0.578
s13l_exp = 0.02225

s12l_err = 0.016
s23l_err = 0.01
s13l_err = 0.00064

deltaCPl_exp = 1.58*np.pi
deltaCPl_err = 0.15*np.pi
'''

#Neutrino oscillation data from NuFit-6.0 
##Normal ordering
dm21_exp = 7.49e-23
dm31_exp = 2.513e-21

dm21_err = 0.19e-23
dm31_err = 0.019e-21

s12l_exp = 0.308
s23l_exp = 0.470
s13l_exp = 0.02215

s12l_err = 0.011
s23l_err = 0.013
s13l_err = 0.00056

deltaCPl_exp = 212*np.pi/180
deltaCPl_err = 26*np.pi/180

##Inverted ordering
'''
dm21_exp = 7.50e-23
dm31_exp = -2.484e-21

dm21_err = 0.19e-23
dm31_err = 0.02e-21

s12l_exp = 0.308
s23l_exp = 0.550
s13l_exp = 0.02231

s12l_err = 0.011
s23l_err = 0.012
s13l_err = 0.00056

deltaCPl_exp = 274*np.pi/180
deltaCPl_err = 22*np.pi/180
'''

#---------------------------------------

#Higgs physics from PDG 06-2026
##Higgs boson
mh_exp = 125.13

mh_err = 0.11


##Higgs couplings
###ATLAS
diph_exp = 1.05
kW_exp = 1.00
kZ_exp = 0.96
kgamma_exp = 0.97
kg_exp = 0.99
kt_exp = 0.99
kb_exp = 0.89
ktau_exp = 0.94
kmu_exp = 1.04
kZgamma_exp = 1.36
k3_exp = (7.2-1.2)/2

diph_err = 0.09
kW_err = 0.05
kZ_err = 0.05
kgamma_err = 0.06
kg_err = 0.06
kt_err = 0.09
kb_err = 0.09
ktau_err = 0.06
kmu_err = 0.23
kZgamma_err = 0.3
k3_err = (7.2+1.2)/2

###CMS 
'''
diph_exp = 1.12
kW_exp = 1.03
kZ_exp = 1.07
kgamma_exp = 1.1
kg_exp = 0.91
kt_exp = 0.92
kb_exp = 0.98
ktau_exp = 0.91
kmu_exp = 1.09
kZgamma_exp = 1.61
k3_exp = (7.02-1.39)/2

diph_err = 0.09
kW_err = 0.06
kZ_err = 0.06
kgamma_err = 0.07
kg_err = 0.06
kt_err = 0.08
kb_err = 0.12
ktau_err = 0.07
kmu_err = 0.20
kZgamma_err = 0.32
k3_err = (7.02+1.39)/2
'''

#---------------------------------------

#Cosmological data from Planck 2018 results 1807.06209
#Dark Matter
Omega_exp = 0.12
Omega_err = 0.001

# Spin-independent direct-detection cross section. The default uncertainty is
# neutral; use CUSTOM_CHI_TARGETS to impose an analysis-specific limit.
sigma_si_exp = 0.0
sigma_si_err = 1.0

#Barion asymmetry
nB_exp = 6.12e-10
YB_exp = 8.69e-11 # Planck 2013 1303.5076

nB_err = 0.04e-10
YB_err = 0.04e-11

#Neutrino mass
nu_sup_exp = 0.12
nu_sup_inf = 0.06

#---------------------------------------
#Electroweak presition observables from PDG 06-2026
MZ_exp = 91.1879
MW_exp = 80.363
ssq_eff_exp = 0.23149
Gamma_Z_exp = 2.4955
Gamma_inv_exp = 498.9e-3

MZ_err = 0.0020
MW_err = 0.008
ssq_eff_err = 0.00022
Gamma_Z_err = 0.0023
Gamma_inv_err = 2.5e-3

#---------------------------------------

#Physical constants from PDG 06-2026
#alpha_e=1/137.035999084 
alpha_e=1/128 #at mZ scale
alpha_s=0.1177 #at mZ scale
vphi = 246.22
GF = 1.1663788e-5

#---------------------------------------
#LFV constraints

BR_mu_egamma_lim = 1.5e-13
BR_mu_egamma_proy = 6e-14
BR_tau_mugamma_lim = 4.4e-8
BR_tau_mugamma_proy = 4.4e-9
BR_tau_egamma_lim = 3.3e-8
BR_tau_egamma_proy = 3.3e-9
BR_mu_3e_proy = 1e-16
CR_muAl_e_proy = 1e-18

# LFV quark charges and nucleon form factors
Qu = 2.0 / 3.0
Qd = -1.0 / 3.0
Qs = -1.0 / 3.0

GVup = 2.0
GVun = 1.0
GVdp = 1.0
GVdn = 2.0
GVsp = 0.0
GVsn = 0.0

GSup = 5.1
GSun = 4.3
GSdp = 4.3
GSdn = 5.1
GSsp = 2.5
GSsn = 2.5

# Aluminum constants for coherent mu-e conversion
ZAl = 13.0
nAl = 14.0
ZeffAl = 11.5
FpAl = 0.64
GammaCaptureAl = 4.64079e-19
