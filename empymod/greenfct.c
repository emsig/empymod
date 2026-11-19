#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<stdbool.h>
#include<complex.h>

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

bool isina(int ab, const int *a, int n);

// Define the reflections function
void reflections(int nfreq, int noff, int nlayer, int nlambda, double *depth, double complex *e_zH, double complex *Gam, int lrec, int lsrc, double complex *Rp, double complex *Rm) ;

// Define the fields function
void fields(int nfreq, int noff, int nlayer, int nlambda, double *depth, double complex *Rp, double complex *Rm, double complex *Gam, int lrec, int lsrc, double zsrc, int ab, bool TM, double complex *Pu, double complex *Pd);

// Define the greenfct function
// Calculate Green's function for TM and TE.
void greenfct(int nfreq, int noff, int nlayer, int nlambda, double zsrc, double zrec, int lsrc, int lrec, double* depth, double complex* etaH, double complex* etaV, double complex* zetaH, double complex* zetaV, double* lambd, int ab, int xdirect, int msrc, int mrec, double complex* GTM, double complex* GTE) 
{
    int g1, g2, g3, r1, r2, n1, n2, n3, nlay;
    int i, ii, iv, iz, dsign, minl, maxl;
    double complex *Gam, *Rp, *Rm, *gamTM, *gamTE;
    double complex *Wu, *Wd, *Pu, *Pd, *green;
    double complex h_div_v, h_times_h, fexp, fact;
    double complex *e_zH;
    double complex *e_zV;
    double complex *z_eH;
    double complex *letaH, *letaV, *lzetaH, *lzetaV, *ptmp, directf;
    double temp, ddepth, dfsign, pmw, l2;
    const int minus_ab[] = {11, 12, 13, 14, 15, 21, 22, 23, 24, 25};

    // GTM/GTE have shape (frequency, offset, lambda).
    // gamTM/gamTE have shape (frequency, offset, layer, lambda):

    g1 = nlambda;
    g2 = g1*nlayer;
    g3 = g2*noff;

    r1=nlambda;
    r2=noff*r1;

    maxl = MAX(lrec, lsrc);
    minl = MIN(lrec, lsrc);
    nlay = (maxl-minl+1);
    n1 = nlambda;
    n2 = n1*nlay;
    n3 = n2*noff;

    //fprintf(stderr,"nfreq=%d noff=%d nlayer=%d nlambda=%d lrec=%d lsrc=%d mrec=%d msrc=%d nlay=%d\n",nfreq, noff, nlayer, nlambda, lrec, lsrc, mrec, msrc, nlay);
    //fprintf(stderr,"r2=%d r1=%d g3=%d g2=%d g1=%d\n",r2, r1, g3, g2, g1);

    gamTM = (double complex *) calloc(nfreq * noff * nlayer * nlambda , sizeof(double complex));
    gamTE = (double complex *) calloc(nfreq * noff * nlayer * nlambda , sizeof(double complex));
    Rp  = (double complex *) calloc(nfreq * noff * nlay * nlambda , sizeof(double complex));
    Rm  = (double complex *) calloc(nfreq * noff * nlay * nlambda , sizeof(double complex));
    Wu  = (double complex *) malloc(nfreq * noff * nlambda * sizeof(double complex));
    Wd  = (double complex *) malloc(nfreq * noff * nlambda * sizeof(double complex));
    Pu  = (double complex *) malloc(nfreq * noff * nlambda * sizeof(double complex));
    Pd  = (double complex *) malloc(nfreq * noff * nlambda * sizeof(double complex));

    //green  = (double complex *) malloc(nfreq * noff * nlambda * sizeof(double complex));
    //fexp    = (double complex *) malloc(nfreq * noff * nlambda * sizeof(double complex));
    letaH   = (double complex *) malloc(nfreq * nlayer * sizeof(double complex));
    letaV   = (double complex *) malloc(nfreq * nlayer * sizeof(double complex));
    lzetaH  = (double complex *) malloc(nfreq * nlayer * sizeof(double complex));
    lzetaV  = (double complex *) malloc(nfreq * nlayer * sizeof(double complex));

// TODO use pointers to assign GTM to green in TM loop
// TODO use pointers to assign Gam to GamTM?

    for (int i = 0; i < nfreq; i++) {
        for (int j = 0; j < nlayer; j++) { //TODO can this be done with pointers and use the - in equations?
            letaH[i*nlayer+j]  = etaH[i*nlayer+j];
            lzetaH[i*nlayer+j] = zetaH[i*nlayer+j];
            letaV[i*nlayer+j]  = etaV[i*nlayer+j];
            lzetaV[i*nlayer+j] = zetaV[i*nlayer+j];
        }
    }
    // Reciprocity switches for magnetic receivers
    if (mrec) {
        if (msrc) { //If src is also magnetic, switch eta and zeta (MM => EE).
            // G^mm_ab(s, r, e, z) = -G^ee_ab(s, r, -z, -e)
            for (int i = 0; i < nfreq; i++) {
                for (int j = 0; j < nlayer; j++) { //TODO can this be done with pointers and use the - in equations?
            //fprintf(stderr," i=%d j=%d  etaH=%e %e\n",  i, j, crealf(etaH[i*nlayer+j]), cimagf(etaH[i*nlayer+j]));
            //fprintf(stderr," i=%d j=%d  etaV=%e %e\n",  i, j, crealf(etaV[i*nlayer+j]), cimagf(etaV[i*nlayer+j]));
                    letaH[i*nlayer+j]  = -zetaH[i*nlayer+j];
                    lzetaH[i*nlayer+j] = -etaH[i*nlayer+j];
                    letaV[i*nlayer+j]  = -zetaV[i*nlayer+j];
                    lzetaV[i*nlayer+j] = -etaV[i*nlayer+j];
                }
            }
        } else { //If src is electric, swap src and rec (ME => EM).
            // G^me_ab(s, r, e, z) = -G^em_ba(r, s, e, z)
            temp = zsrc;
            zsrc = zrec;
            zrec = temp;
            temp = lsrc;
            lsrc = lrec;
            lrec = temp;
        }
    }

    for (int TM = 0; TM < 2; TM++) {
        // Continue if Green's function not required
        if (TM && (ab == 16 || ab == 26)) {
            continue;
        } else if (!TM && (ab == 13 || ab == 23 || ab == 31 || ab == 32 || ab == 33 || ab == 34 || ab == 35)) {
            continue;
        }

        // Define eta/zeta depending if TM or TE
        if (TM) {
            e_zH = letaH;
            e_zV = letaV;
            z_eH = lzetaH;
            green = GTM;
            Gam = gamTM;
//TODO set pointer for green and Gam
        } else {
            e_zH = lzetaH;
            e_zV = lzetaV;
            z_eH = letaH;
            green = GTE;
            Gam = gamTE;
        }

        // Uppercase gamma
        for (int i = 0; i < nfreq; i++) {
            for (int iii = 0; iii < nlayer; iii++) {
                h_div_v = e_zH[i*nlayer+iii] / e_zV[i*nlayer+iii];
                h_times_h = z_eH[i*nlayer+iii] * e_zH[i*nlayer+iii];
            //fprintf(stderr," i=%d iii=%d  h_div_v=%e %e\n",  i, iii, crealf(h_div_v), cimagf(h_div_v));
            //fprintf(stderr," i=%d iii=%d  h_times_h=%e %e\n",  i, iii, crealf(h_times_h), cimagf(h_times_h));
                for (int ii = 0; ii < noff; ii++) {
                    for (int iv = 0; iv < nlambda; iv++) {
                        l2 = lambd[ii*nlambda+iv] * lambd[ii*nlambda+iv];
            //fprintf(stderr," ii=%d iv=%d  l2=%e %e\n",  ii, iv, crealf(l2), cimagf(l2));
                        Gam[i*g3+ii*g2+iii*g1+iv] = csqrt(h_div_v * l2 + h_times_h);
            //fprintf(stderr,"Gam i=%d ii=%d iii=%d iv=%d %e %e\n",  i, ii, iii, iv, crealf(Gam[i*g3+ii*g2+iii*g1+iv]), cimagf(Gam[i*g3+ii*g2+iii*g1+iv]));
                    }
                }
            }
        }

        // Gamma in receiver layer

        // Reflection (coming from below (Rp) and above (Rm) rec)
        if (nlayer > 1) {
            // TODO set Rp Rm to zero if needed
            reflections(nfreq, noff, nlayer, nlambda, depth, e_zH, Gam, lrec, lsrc, Rp, Rm);
            // Field propagators
            // (Up- (Wu) and downgoing (Wd), in rec layer); Eq 74

            if (lrec != nlayer - 1) {
                ddepth = depth[lrec + 1] - zrec;
                for (int i = 0; i < nfreq; i++) {
                    for (int ii = 0; ii < noff; ii++) {
                        for (int iv = 0; iv < nlambda; iv++) {
                            Wu[i*r2+ii*r1+iv] = cexp(-Gam[i*g3+ii*g2+lrec*g1+iv] * ddepth);
                        }
                    }
                }
            }

            if (lrec != 0) {
                ddepth = zrec - depth[lrec];
                for (int i = 0; i < nfreq; i++) {
                    for (int ii = 0; ii < noff; ii++) {
                        for (int iv = 0; iv < nlambda; iv++) {
                            Wd[i*r2+ii*r1+iv] = cexp(-Gam[i*g3+ii*g2+lrec*g1+iv] * ddepth);
                        }
                    }
                }
            }

            // Field at rec level (coming from below (Pu) and above (Pd) rec)
            // set Pu Pd to zero
            //memset(Pu,0,nfreq*noff*nlambda*sizeof(double complex));
            //memset(Pd,0,nfreq*noff*nlambda*sizeof(double complex));
            fields(nfreq, noff, nlayer, nlambda, depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM, Pu, Pd);
        }

        // Green's functions

        if (lsrc == lrec) { //Rec in src layer; Eqs 108, 109, 110, 117, 118, 122
            // Green's function depending on <ab>
            // (If only one layer, no reflections/fields)
            if (nlayer > 1 && (ab == 13 || ab == 23 || ab == 31 || ab == 32 || ab == 14 || ab == 24 || ab == 15 || ab == 25)) {
                for (int i = 0; i < nfreq; i++) {
                    for (int ii = 0; ii < noff; ii++) {
                        for (int iv = 0; iv < nlambda; iv++) {
                            green[i*r2+ii*r1+iv] = Pu[i*r2+ii*r1+iv]*Wu[i*r2+ii*r1+iv];
                            green[i*r2+ii*r1+iv] -= Pd[i*r2+ii*r1+iv]*Wd[i*r2+ii*r1+iv];
                        }
                    }
                }
            } else if (nlayer > 1) {
                for (int i = 0; i < nfreq; i++) {
                    for (int ii = 0; ii < noff; ii++) {
                        for (int iv = 0; iv < nlambda; iv++) {
                            green[i*r2+ii*r1+iv] = Pu[i*r2+ii*r1+iv]*Wu[i*r2+ii*r1+iv];
                            green[i*r2+ii*r1+iv] += Pd[i*r2+ii*r1+iv]*Wd[i*r2+ii*r1+iv];
                        }
                    }
                }
            }

            // Direct field, if it is computed in the wavenumber domain
            if (!xdirect) {
                ddepth = abs(zsrc - zrec);
                if ((zrec - zsrc) < 0 ) dsign = -1;
                else dsign = 1;

                // Swap TM for certain <ab>
                dfsign = 1;
                if (TM && isina(ab, minus_ab, 10)) {
                    dfsign = -1;
                }
                //# Multiply by zrec-zsrc-sign for certain <ab>
                if ((ab == 11 || ab == 12 || ab == 13 || ab == 14 || ab == 15 || ab == 21 || ab == 22 || ab == 23 || ab == 24 || ab == 25)) {
                    dfsign *= dsign;
                }

                for (int i = 0; i < nfreq; i++) {
                    for (int ii = 0; ii < noff; ii++) {
                        for (int iv = 0; iv < nlambda; iv++) {
                            // Direct field
                            directf = dfsign*cexp(-Gam[i*g3+ii*g2+lrec*g1+iv]*ddepth);

                            // Add direct field to Green's function
                            green[i*r2+ii*r1+iv] += directf;
                        }
                    }
                }

                // Implementation
            }
        } else {
            // Calculate exponential factor
            if (lrec == nlayer-1) { 
                ddepth = 0;
            }
            else {
                ddepth = depth[lrec+1] - depth[lrec];
            }

/* replaced by scalar in loops
            for (int i = 0; i < nfreq; i++) {
                for (int ii = 0; ii < noff; ii++) {
                    for (int iv = 0; iv < nlambda; iv++) {
                        fexp[i*r2+ii*r1+iv] = cexp(-Gam[i*g3+ii*g2+lrec*g1+iv]*ddepth);
                    }
                }
            }
*/

            // Sign-switch for Green calculation
            if (TM && isina(ab, minus_ab, 10)) {
                pmw = -1;
            }
            else {
                pmw = 1;
            }

            if (lrec < lsrc) {  // Rec above src layer: Pd not used
                //           Eqs 89-94, A18-A23, B13-B15
                for (int i = 0; i < nfreq; i++) {
                    for (int ii = 0; ii < noff; ii++) {
                        for (int iv = 0; iv < nlambda; iv++) {
                            fexp = cexp(-Gam[i*g3+ii*g2+lrec*g1+iv]*ddepth);
                            green[i*r2+ii*r1+iv] = Pu[i*r2+ii*r1+iv]*(
                                    Wu[i*r2+ii*r1+iv] + pmw*Rm[i*n3+ii*n2+0*n1+iv] *
                                    fexp*Wd[i*r2+ii*r1+iv]);
                        }
                    }
                }
            }
            else if (lrec > lsrc) {  // rec below src layer: Pu not used
                //                Eqs 97-102 A26-A30, B16-B18
                for (int i = 0; i < nfreq; i++) {
                    for (int ii = 0; ii < noff; ii++) {
                        for (int iv = 0; iv < nlambda; iv++) {
                            fexp = cexp(-Gam[i*g3+ii*g2+lrec*g1+iv]*ddepth);
                            green[i*r2+ii*r1+iv] = Pd[i*r2+ii*r1+iv]*(
                                    pmw*Wd[i*r2+ii*r1+iv] +
                                    Rp[i*n3+ii*n2+abs(lsrc-lrec)*n1+iv] *
                                    fexp*Wu[i*r2+ii*r1+iv]);
            //fprintf(stderr,"Rp*fexpWu i=%d ii=%d iv=%d %e %e\n",  i, ii, iv, crealf(Rp[i*g3+ii*g2+abs(lsrc-lrec)*g1+iv]*fexp*Wu[i*r2+ii*r1+iv]), cimagf(Rp[i*g3+ii*g2+abs(lsrc-lrec)*g1+iv]*fexp*Wu[i*r2+ii*r1+iv]));
            //fprintf(stderr,"Rp i=%d ii=%d iv=%d %e %e\n",  i, ii, iv, crealf(Rp[i*n3+ii*n2+abs(lsrc-lrec)*n1+iv]), cimagf(Rp[i*n3+ii*n2+abs(lsrc-lrec)*n1+iv]));
                        }
                    }
                }
            }
        }
        // Store in corresponding variable
        // TODO: => done to check
        //if (TM) {
            //gamTM = Gam;
            //memcpy(GTM, green, nfreq*noff*nlambda*sizeof(double complex));
            //gamTM, GTM = Gam, green
        //}
        //else{
            //gamTE = Gam;
            //memcpy(GTE, green, nfreq*noff*nlambda*sizeof(double complex));
            //gamTE, GTE = Gam, green
        //}

    } // end of TM loop

    // ** AB-SPECIFIC FACTORS AND CALCULATION OF PTOT'S
    // These are the factors inside the integrals
    // Eqs 105-107, 111-116, 119-121, 123-128
    //
    
    if (ab == 11 || ab == 12 || ab == 21 || ab == 22) {
        for (int i = 0; i < nfreq; ++i) {
            fact = 1.0 / letaH[i*nlayer+lrec];
            for (int ii = 0; ii < noff; ++ii) {
                for (int iv = 0; iv < nlambda; ++iv) {
            //fprintf(stderr,"GTM i=%d ii=%d iv=%d %e %e\n",  i, ii, iv, crealf(GTM[i*r2+ii*r1+iv]), cimagf(GTM[i*r2+ii*r1+iv]));
                    GTM[i*r2+ii*r1+iv] *= fact * gamTM[i*g3+ii*g2+lrec*g1+iv];
                    GTE[i*r2+ii*r1+iv] *= lzetaH[i*nlayer+lsrc] / gamTE[i*g3+ii*g2+lsrc*g1+iv];
            //fprintf(stderr,"gamTE i=%d ii=%d iv=%d %e %e\n",  i, ii, iv, crealf(gamTE[i*g3+ii*g2+lsrc*g1+iv]), cimagf(gamTE[i*g3+ii*g2+lsrc*g1+iv]));
                }
            }
        }
    } else if (ab == 14 || ab == 15 || ab == 24 || ab == 25) {
        for (int i = 0; i < nfreq; ++i) {
            fact = letaH[i*nlayer+lsrc] / letaH[i*nlayer+lrec];
            for (int ii = 0; ii < noff; ++ii) {
                for (int iv = 0; iv < nlambda; ++iv) {
                    GTM[i*r2+ii*r1+iv] *= fact * gamTM[i*g3+ii*g2+lrec*g1+iv];
                    GTM[i*r2+ii*r1+iv] /= gamTM[i*g3+ii*g2+lsrc*g1+iv];
                }
            }
        }
    } else if (ab == 13 || ab == 23) {
        memset(GTE,0,nfreq*noff*nlambda*sizeof(double complex));
        for (int i = 0; i < nfreq; ++i) {
            fact = letaH[i*nlayer+lsrc] / letaH[i*nlayer+lrec] / letaV[i*nlayer+lsrc];
            for (int ii = 0; ii < noff; ++ii) {
                for (int iv = 0; iv < nlambda; ++iv) {
                    GTM[i*r2+ii*r1+iv] *= -fact * gamTM[i*g3+ii*g2+lrec*g1+iv];
                    GTM[i*r2+ii*r1+iv] /= gamTM[i*g3+ii*g2+lsrc*g1+iv];
                }
            }
        }
    } else if (ab == 31 || ab == 32) {
        memset(GTE,0,nfreq*noff*nlambda*sizeof(double complex));
        for (int i = 0; i < nfreq; ++i) {
            fact= 1.0/letaV[i*nlayer+lrec];
            for (int ii = 0; ii < noff; ++ii) {
                for (int iv = 0; iv < nlambda; ++iv) {
                    GTM[i*r2+ii*r1+iv] *= fact;
                }
            }
        }
    } else if (ab == 34 || ab == 35) {
        memset(GTE,0,nfreq*noff*nlambda*sizeof(double complex));
        for (int i = 0; i < nfreq; ++i) {
            fact = letaH[i*nlayer+lsrc]/letaV[i*nlayer+lrec];
            for (int ii = 0; ii < noff; ++ii) {
                for (int iv = 0; iv < nlambda; ++iv) {
                    GTM[i*r2+ii*r1+iv] *= fact/gamTM[i*g3+ii*g2+lsrc*g1+iv];
                }
            }
        }
    } else if (ab == 16 || ab == 26) {
        memset(GTM,0,nfreq*noff*nlambda*sizeof(double complex));
        for (int i = 0; i < nfreq; ++i) {
            fact = lzetaH[i*nlayer+lsrc]/lzetaV[i*nlayer+lsrc];
            for (int ii = 0; ii < noff; ++ii) {
                for (int iv = 0; iv < nlambda; ++iv) {
                    GTE[i*r2+ii*r1+iv] *= fact/gamTE[i*g3+ii*g2+lsrc*g1+iv];
                }
            }
        }

    } else if (ab == 33) {
        memset(GTE,0,nfreq*noff*nlambda*sizeof(double complex));
        for (int i = 0; i < nfreq; ++i) {
            fact = letaH[i*nlayer+lsrc]/letaV[i*nlayer+lsrc]/letaV[i*nlayer+lrec];
            for (int ii = 0; ii < noff; ++ii) {
                for (int iv = 0; iv < nlambda; ++iv) {
                    GTM[i*r2+ii*r1+iv] *= fact/gamTM[i*g3+ii*g2+lsrc*g1+iv];
                }
            }
        }
    }

    free(letaH);
    free(letaV);
    free(lzetaH);
    free(lzetaV);
    free(Rp);
    free(Rm);
    free(Wu);
    free(Wd);
    free(Pu);
    free(Pd);
    free(gamTM);
    free(gamTE);

    // Return Green's functions GTM and GTE
    return;

}


