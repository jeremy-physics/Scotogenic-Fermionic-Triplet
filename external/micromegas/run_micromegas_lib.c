/* run_micromegas_lib.c: shared-library interface for ELBAPH2.0.
 *
 * This file preserves the legacy microELBAPH micrOMEGAs calculation and adds
 * a small status-returning wrapper for the Python ctypes backend.
 */
#include "../include/micromegas.h"
#include "../include/micromegas_aux.h"
#include "lib/pmodel.h"
#include <math.h>
#include <stdio.h>

#define ELBAPH_MICRO_N_INPUTS 18

int is_initialized = 0;

void initialize_micromegas() {
    if (!is_initialized) {
        ForceUG=0; VZdecay=0; VWdecay=0;
        if(readVar("data_run.par") == 0) {
            /* Success */
        }
        is_initialized = 1;
    }
}

void get_observables(double* inputs, double* results) {
    int k = 0;

    /* 1. Assign Parameters */
    assignValW("MR",  inputs[k++]);
    assignValW("MN1", inputs[k++]);
    assignValW("MN2", inputs[k++]);
    assignValW("MN3", inputs[k++]);

    assignValW("la2", inputs[k++]);
    assignValW("la3", inputs[k++]);
    assignValW("la4", inputs[k++]);
    assignValW("la5", inputs[k++]);

    assignValW("YN11",  inputs[k++]);
    assignValW("YN12r", inputs[k++]);
    assignValW("YN12i", inputs[k++]);
    assignValW("YN13",  inputs[k++]);

    assignValW("YN21", inputs[k++]);
    assignValW("YN22", inputs[k++]);
    assignValW("YN23", inputs[k++]);

    assignValW("YN31", inputs[k++]);
    assignValW("YN32", inputs[k++]);
    assignValW("YN33", inputs[k++]);

    /* 2. Calculation */
    results[0] = -1.0;
    results[1] = 0.0;

    int err;
    char cdmName[10];
    double Xf;

    if(sortOddParticles(cdmName) != 0) return;

    double Omega = darkOmega(&Xf, 1, 1.E-4, &err);
    if(err) return;

    double sigma_SI_p = 0.0;
    if(Omega > 1e-9) {
        double pA0[2], pA5[2], nA0[2], nA5[2];
        double Nmass = 0.939;
        nucleonAmplitudes(CDM[1], pA0, pA5, nA0, nA5);

        double mu = Nmass * Mcdm / (Nmass + Mcdm);
        double SCcoeff = 4 / M_PI * 3.8937966E8 * pow(mu, 2.);
        sigma_SI_p = SCcoeff * pA0[0] * pA0[0];
    }

    results[0] = Omega;
    results[1] = sigma_SI_p;
}

int run_micromegas(const double *inputs, int n_inputs, double *omega, double *sigmaSI) {
    if(inputs == NULL || omega == NULL || sigmaSI == NULL) {
        return 10;
    }
    if(n_inputs != ELBAPH_MICRO_N_INPUTS) {
        return 11;
    }

    double local_inputs[ELBAPH_MICRO_N_INPUTS];
    double results[2];
    int i;

    for(i = 0; i < ELBAPH_MICRO_N_INPUTS; i++) {
        local_inputs[i] = inputs[i];
    }

    initialize_micromegas();
    get_observables(local_inputs, results);

    *omega = results[0];
    *sigmaSI = results[1];

    if(results[0] < 0.0) {
        return 12;
    }

    return 0;
}
