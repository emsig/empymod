#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<stdbool.h>
#include<complex.h>

// Function prototypes
void greenfct(int nfreq, int noff, int nlayer, int nlambda, double zsrc, double zrec, int lsrc, int lrec, double* depth, double complex* etaH, double complex* etaV, double complex* zetaH, double complex* zetaV, double* lambd, int ab, int xdirect, int msrc, int mrec, double complex* GTM, double complex* GTE);

void wavenumber(int nfreq, int noff, int nlayer, int nlambda, 
                double zsrc, double zrec, int lsrc, int lrec, double *depth,
                double complex *etaH, double complex *etaV, double complex *zetaH, double complex *zetaV,
                double *lambd, int ab, int xdirect, int msrc, int mrec, 
                double complex *PJ0, double complex *PJ1, double complex *PJ0b) {
    int g1, g2, g3, r1, r2;
    int i, ii, iv;
    double fourpi, eightpi, sign, dlambd, tlambd;
    double complex *PTM, *PTE, Ptot;

    // Calculate Green's functions
    g1 = nlambda;
    g2 = g1*nlayer;
    g3 = g2*noff;

    r1=nlambda;
    r2=noff*r1;

    PTM  = (double complex *) calloc(nfreq * noff * nlambda , sizeof(double complex));
    PTE  = (double complex *) calloc(nfreq * noff * nlambda , sizeof(double complex));

    greenfct(nfreq, noff, nlayer, nlambda, zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
             ab, xdirect, msrc, mrec, PTM, PTE);

    // Pre-allocate output
    if (ab == 11 || ab == 22 || ab == 24 || ab == 15 || ab == 33) {
        PJ0 = PJ0;
    } else {
        PJ0 = NULL;
    }
    if (ab == 11 || ab == 12 || ab == 21 || ab == 22 || ab == 14 || ab == 24 ||
        ab == 15 || ab == 25) {
        PJ0b = PJ0b;
    } else {
        PJ0b = NULL;
    }
    if (ab != 33) {
        PJ1 = PJ1;
    } else {
        PJ1 = NULL;
    }


    //fourpi = 1.0 /(4 * 3.14159265358979323846);
    fourpi = 1.0 /(4 * M_PI);
    // If rec is magnetic switch sign (reciprocity MM/ME => EE/EM)
    if (mrec) {
        sign = -1.0;
    } else {
        sign = 1.0;
    }

    // Group into PJ0 and PJ1 for J0/J1 Hankel Transform
    if (ab == 11 || ab == 12 || ab == 21 || ab == 22 || ab == 14 || ab == 24 ||
        ab == 15 || ab == 25) {
        if (ab == 14 || ab == 22) {
            sign *= -1;
        }

        for (i = 0; i < nfreq; i++) {
            for (ii = 0; ii < noff; ii++) {
                for (iv = 0; iv < nlambda; iv++) {
                    Ptot = (PTM[i*r2+ii*r1+iv] + PTE[i*r2+ii*r1+iv]) * fourpi;
                    PJ0b[i*r2+ii*r1+iv] = sign * 0.5 * Ptot * lambd[ii*nlambda+iv];
                    PJ1[i*r2+ii*r1+iv] = -1.0*sign * Ptot;
                }
            }
        }

        if (ab == 11 || ab == 22 || ab == 24 || ab == 15) {
            if (ab == 22 || ab == 24) {
                sign *= -1;
            }

            eightpi = sign / (8 * M_PI);
            for (i = 0; i < nfreq; i++) {
                for (ii = 0; ii < noff; ii++) {
                    for (iv = 0; iv < nlambda; iv++) {
                        PJ0[i*r2+ii*r1+iv] = (PTM[i*r2+ii*r1+iv] - PTE[i*r2+ii*r1+iv])*lambd[ii*nlambda+iv] * eightpi;
                    }
                }
            }
        }

    } else if (ab == 13 || ab == 23 || ab == 31 || ab == 32 || ab == 34 ||
               ab == 35 || ab == 16 || ab == 26) {
        if (ab == 34 || ab == 26) {
            sign *= -1;
        }
        for (i = 0; i < nfreq; i++) {
            for (ii = 0; ii < noff; ii++) {
                for (iv = 0; iv < nlambda; iv++) {
                    dlambd = lambd[ii*nlambda+iv] * lambd[ii*nlambda+iv];
                    Ptot = (PTM[i*r2+ii*r1+iv] + PTE[i*r2+ii*r1+iv]) * fourpi;
                    PJ1[i*r2+ii*r1+iv] = sign * Ptot * dlambd;
                }
            }
        }
    } else if (ab == 33) {
        for (i = 0; i < nfreq; i++) {
            for (ii = 0; ii < noff; ii++) {
                for (iv = 0; iv < nlambda; iv++) {
                    tlambd = lambd[ii*nlambda+iv] * lambd[ii*nlambda+iv] * lambd[ii*nlambda+iv];
                    Ptot = (PTM[i*r2+ii*r1+iv] + PTE[i*r2+ii*r1+iv]) * fourpi;
                    PJ0[i*r2+ii*r1+iv] = sign * Ptot * tlambd;
                }
            }
        }
    }
    free(PTM);
    free(PTE);
    return;
}

