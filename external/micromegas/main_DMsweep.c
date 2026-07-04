#include "../include/micromegas.h"
#include "../include/micromegas_aux.h"
#include "lib/pmodel.h"
#include <time.h> 
#include <stdlib.h>
#include <math.h>

double vevVal;

double get_random_mag_sign(double min_mag, double max_mag) {
    double mag = min_mag + (max_mag - min_mag) * ((double)rand() / RAND_MAX);
    int sign = (rand() % 2 == 0) ? 1 : -1;
    return mag * sign;
}

double get_random_range(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

int main(int argc, char** argv)
{
    int err;
    char cdmName[10];
    
    // Inicializar semilla
    srand(time(NULL)); 

    ForceUG=0; 
    VZdecay=0; VWdecay=0;

    printf("\n=== INICIANDO ESCANEO ===\n");

    // 1. CARGA DE PARÁMETROS BASE
    err = readVar("data_DM.par"); 
    if(err == -1) {
        printf("ERROR: No se pudo abrir 'data.par'.\n");
        exit(1);
    }
    printf("Parámetros base cargados.\n");

    // 2. PREPARACIÓN DE SALIDAS
    FILE *fd_full = fopen("DMscan_full.dat", "w");
    fprintf(fd_full, "# MR "
                   "la2 la3 la4 la5 " 
                   "lam_port "
                   "Omega SigmaSI_p\n"); 
    
    FILE *fd_ok = fopen("DMscan_ok.dat", "w");
    fprintf(fd_ok, "# MR "
                   "la2 la3 la4 la5 " 
                   "lam_port "
                   "Omega SigmaSI_p\n"); 

    FILE *fd_valid = fopen("DMscan_valid.dat", "w");
    fprintf(fd_valid, "# MR "
                   "la2 la3 la4 la5 " 
                   "lam_port "
                   "Omega SigmaSI_p Pval_DD\n"); 

    // Variables internas micrOMEGAs
    double Omega, Xf;
    int fast = 1;         
    double Beps = 1.E-4;  
    char* expName;
    double pval;          

    // Variables para Detección Directa (Amplitudes y Sección Eficaz)
    double pA0[2], pA5[2], nA0[2], nA5[2]; // Arrays para amplitudes 
    double Nmass = 0.939; // Masa del nucleón en GeV
    double SCcoeff;       // Coeficiente de masa reducida
    double sigma_SI_p;    // Sección eficaz SI protón en pb

    // Variables a Escanear
    double MR;
    double la2, la3, la4, la5;
    double lam_port;

    // Contadores
    int Ntot = 1.E4;
    int n_full = 0;
    int n_ok = 0;
    int n_valid = 0;

    printf("Escaneando %d puntos...\n", Ntot);

    // 3. BUCLE DE ESCANEO
    for(int i = 1; i <= Ntot; i++)
    {
        // A. GENERACIÓN ALEATORIA
        MR = get_random_range(200, 5000);

        la2 = get_random_range(1e-3, 1);
        la3 = get_random_range(-1, -1e-3);
        la4 = get_random_range(-1, -5e-2);
        la5 = get_random_range(-1, -1e-7);

        // B. ASIGNACIÓN AL MODELO
        assignValW("MR", MR);

        assignValW("la2", la2);
        assignValW("la3", la3);
        assignValW("la4", la4);
        assignValW("la5", la5);

        vevVal = findValW("vevVal");
        
        lam_port = vevVal*(la3 + la4 + la5);

        // C. CÁLCULO DE ESPECTRO
        err = sortOddParticles(cdmName);
        if(err) { continue; } 

        // D. CÁLCULO DE DENSIDAD RELIQUIA
        Omega = darkOmega(&Xf, fast, Beps, &err);

        if(err != 0) {
        printf("Advertencia: Error numérico en punto %d (err=%d). Saltando.\n", i, err);
        continue; // Salta al siguiente punto del bucle for
}

        // E. LÓGICA DE FILTRADO
        //if(Omega > 1.0E-15 && Omega < 1.0E3)
        if(Omega > 0) 
        {
            // --- CÁLCULO DE SIGMA SI (Nuevo) ---
            // 1. Obtener amplitudes para el candidato CDM[1]
            nucleonAmplitudes(CDM[1], pA0, pA5, nA0, nA5);
            
            // 2. Calcular factor cinemático (Mcdm es variable global de micrOMEGAs)
            // 3.8937966E8 convierte GeV^-2 a pb
            SCcoeff = 4 / M_PI * 3.8937966E8 * pow(Nmass * Mcdm / (Nmass + Mcdm), 2.);
            
            // 3. Calcular Sección Eficaz Protón Spin-Independent
            sigma_SI_p = SCcoeff * pA0[0] * pA0[0];

            // --- GUARDAR EN SCAN_OK ---
            fprintf(fd_full, "%.4E "
                           "%.4E %.4E %.4E %.4E "
                           "%.4E "
                           "%.4E %.4E\n",  // Agregado %.4E para Sigma
                    MR,
                    la2, la3, la4, la5,
                    lam_port,
                    Omega, sigma_SI_p); // Guardamos Sigma
            
            n_full++; 
            
            // --- GUARDAR EN SCAN_OK_valid (Planck) ---
            if(Omega > 1.0E-3 && Omega < 0.15)
            {
                // Calcular p-value experimental

                fprintf(fd_ok, "%.4E "
                               "%.4E %.4E %.4E %.4E "
                               "%.4E "
                               "%.4E %.4E\n", // Dos %.4E al final (Sigma y Pval)
                        MR,
                        la2, la3, la4, la5,
                        lam_port,
                        Omega, sigma_SI_p); // Guardamos Sigma y Pval
                
                n_ok++; 
            }
        
                // --- GUARDAR EN SCAN_OK_valid (Planck) ---
                if (Omega >= 0.1164 && Omega <= 0.1236) 
                {
                    // Calcular p-value experimental
                    pval = DD_pval(AllDDexp, Maxwell, &expName);

                    fprintf(fd_valid, "%.4E "
                                "%.4E %.4E %.4E %.4E "
                                "%.4E "
                                "%.4E %.4E %.4E\n", // Dos %.4E al final (Sigma y Pval)
                            MR,
                            la2, la3, la4, la5,
                            lam_port,
                            Omega, sigma_SI_p, pval); // Guardamos Sigma y Pval
                    
                    n_valid++; 
                }
        }
        if(i % 500 == 0) {
            printf("Progreso: %d/%d | Full: %d | OK: %d | Planck: %d | Om=%.2e | Sigma=%.2e\n", 
                   i, Ntot, n_full , n_ok, n_valid, Omega, sigma_SI_p);
        }
    }

    fclose(fd_full);
    fclose(fd_ok);
    fclose(fd_valid);
    
    printf("\n=== ESCANEO FINALIZADO ===\n");
    printf("Puntos en 'DMscan_full.dat': %d\n", n_full);
    printf("Puntos en 'DMscan_ok.dat': %d\n", n_ok);
    printf("Puntos en 'DMscan_valid.dat': %d\n", n_valid);
    
    #ifdef CLEAN
        system("rm -f HB.* hs.* debug_channels.txt Key.dat Lilith_* particles.py* smodels.*");
    #endif

    return 0;
}