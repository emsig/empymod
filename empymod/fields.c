#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include<assert.h>
#include<stdbool.h>
#include<complex.h>

bool isina(int ab, const int *a, int n) {
    int i;
    bool ok;

    ok=false;
    for (i=0;i<n;i++) {
        if (a[i] == ab) ok=true;
    }
    return ok;
}
    
// TODO complex *Pu, complex *Pd initialised=0

void fields(int nfreq, int noff, int nlayer, int nlambda, double *depth, double complex *Rp, double complex *Rm, double complex *Gam, int lrec, int lsrc, double zsrc, int ab, bool TM, double complex *Pu, double complex *Pd)
{
    double ds, dm, dp, ftmp, ddepth; 
    int nlsr, rsrcl, isr, last, pm, mupm, i, ii, iv, iz, iii, iiii;
    int pup, up, itmp,izstart, izend;
    int g1, g2, g3, n1, n2, m1, m2, m3;
    bool first_layer, last_layer, plus;
    double complex *P, *Rmp, *Rpm;
    double complex tRmp, tRpm, p1, p2, p3, tiRpm, iRpm, iRmp, piGam, piGam2, tiGam;

    //fprintf(stderr,"nfreq=%d noff=%d nlayer=%d nlambda=%d lrec=%d lsrc=%d zsrc=%f ab=%d TM=%d\n",nfreq, noff, nlayer, nlambda, lrec, lsrc, zsrc, ab, TM);

/*
    fprintf(stderr,"d=%f %f %f\n",depth[0], depth[1], depth[2]);

    for (iiii=0; iiii<1; iiii++) {
    for (iii=0; iii<2; iii++) {
    for (ii=0; ii<3; ii++) {
    for (i=0; i<4; i++) {
    fprintf(stderr,"Gam[%d][%d][%d][%d]=%e %e\n", iiii, iii, ii, i, crealf(Gam[iiii*g3+iii*g2+ii*g1+i]), cimagf(Gam[iiii*g3+iii*g2+ii*g1+i]));
    }
    }
    }
    }
*/

    // Variables
    nlsr = abs(lsrc-lrec)+1;  // nr of layers btw and incl. src and rec layer
    rsrcl = 0;  // src-layer in reflection (Rp/Rm), first if down
    izstart=2;
    izend=nlsr;
    isr = lsrc;
    last = nlayer-1;

// Dimension of the different arrays
    g1 = nlambda;
    g2 = g1*nlayer;
    g3 = g2*noff;

    n1 = nlambda;
    n2 = n1*noff;

    m1 = nlambda;
    m2 = m1*nlsr;
    m3 = m2*noff;

    // Booleans if src in first or last layer; swapped if up=True
    if (lsrc == 0) first_layer=1;
    else first_layer=0;

    if (lsrc == nlayer-1) last_layer=1;
    else last_layer=0;

    // Depths; dp and dm are swapped if up=True
    if (lsrc != nlayer-1) {
        ds = depth[lsrc+1]-depth[lsrc];
        dp = depth[lsrc+1]-zsrc;
    }
    dm = zsrc-depth[lsrc];

    // Boolean if plus or minus has to be calculated
    const int plusset[] = {13, 23, 33, 14, 24, 34, 15, 25, 35};
    if (TM){
        plus = isina(ab,plusset,9);
    }
    else{
        plus = !isina(ab,plusset,9);
    }

    // Sign-switches
    // + if plus=True, - if plus=False
    if (plus) pm=1;
    else pm=-1;
    pup = -1;  // + if up=True,   - if up=False
    mupm = 1;  // + except if up=True and plus=False

    // Calculate down- and up-going fields
    for (up=0;up<=1;up++) { 

        // No upgoing field if rec is in last layer or below src
        if (up==1 && (lrec == nlayer-1 || lrec > lsrc)) {
            memset(Pu,0,nfreq*noff*nlambda*sizeof(double complex));
            continue;
        }
        // No downgoing field if rec is in first layer or above src
        if (up==0 && (lrec==0 || lrec < lsrc)) {
            memset(Pd,0,nfreq*noff*nlambda*sizeof(double complex));
            continue;
        }

        // Swaps if up=True
        if (up==1){
            if (!last_layer) {
                ftmp=dp;
                dp=dm;
                dm=ftmp;
            }
            else{
                dp = dm;
            }
            // reference
            Rmp = Rp;
            Rpm = Rm;
            itmp=first_layer;
            first_layer=last_layer; 
            last_layer=itmp;
            rsrcl = nlsr-1;  // src-layer in refl. (Rp/Rm), last (nlsr-1) if up
            izstart=0;
            izend=nlsr-2;
            isr = lrec;
            last = 0;
            pup = 1;
            if (!plus) mupm = -1;
            P = Pu;
        }
        else{
            // reference
            Rmp = Rm;
            Rpm = Rp;
            P = Pd;
        }

        // Calculate Pu+, Pu-, Pd+, Pd-
        if (lsrc == lrec) { // rec in src layer; Eqs  81/82, A-8/A-9
            if (last_layer) {  // If src/rec are in top (up) or bottom (down) layer
                for (i=0;i<nfreq;i++) { 
                    for (ii=0;ii<noff;ii++) { 
#pragma ivdep
                        for (iv=0;iv<nlambda;iv++) { 
                            //tRmp = Rmp[i*m3+ii*m2+0*m1+iv];
                            //tiGam = Gam[i*g3+ii*g2+lsrc*g1+iv];
                            //P[i*n2+ii*n1+iv] = tRmp*cexp(-tiGam*dm);
                            P[i*n2+ii*n1+iv] = Rmp[i*m3+ii*m2+0*m1+iv]*cexp(-Gam[i*g3+ii*g2+lsrc*g1+iv]*dm);
                        }
                    }
                }
            }
            else {           // If src and rec are in any layer in between
//fprintf(stderr,"nfreq=%d noff=%d nlayer=%d nlambda=%d\n",nfreq, noff, nlayer, nlambda);
                for (i=0;i<nfreq;i++) { 
                    for (ii=0;ii<noff;ii++) {
#pragma ivdep
                        for (iv=0;iv<nlambda;iv++) {
                            tRpm = Rpm[i*m3+ii*m2+0*m1+iv];
                            tRmp = Rmp[i*m3+ii*m2+0*m1+iv];
                            tiGam = Gam[i*g3+ii*g2+lsrc*g1+iv];
                            p1 = cexp(-tiGam * dm);
                            p2 = pm * tRpm * cexp(-tiGam * (ds + dp));
                            p3 = 1.0 - tRmp * tRpm * cexp(-2 * tiGam * ds);
                            P[i*n2+ii*n1+iv] = ((p1 + p2) * (tRmp/p3));

                            //p1 = cexp(-Gam[i*g3+ii*g2+lsrc*g1+iv]*dm);
                            //p2 = pm*tRpm*cexp(-Gam[i*g3+ii*g2+lsrc*g1+iv]*(ds+dp));
                            //p3 = 1 - tRmp * tRpm * cexp(-2*Gam[i*g3+ii*g2+lsrc*g1+iv]*ds);
/*
fprintf(stderr,"P[%d][%d][%d]\n",  i, ii, iv);
    fprintf(stderr,"p1=%e %e\n",  crealf(p1), cimagf(p1));
    fprintf(stderr,"p2=%e %e\n",  crealf(p2), cimagf(p2));
    fprintf(stderr,"p3=%e %e\n",  crealf(p3), cimagf(p3));
    fprintf(stderr,"tRmp=%e %e\n",  crealf(tRmp), cimagf(tRmp));
    fprintf(stderr,"tRpm=%e %e\n",  crealf(tRpm), cimagf(tRpm));
    fprintf(stderr,"Rmp[%d][%d][%d]=%e %e\n", i, ii, iv, crealf(Rmp[i*n2+ii*n1+iv]), cimagf(Rmp[i*n2+ii*n1+iv]));
    fprintf(stderr,"Rpm[%d][%d][%d]=%e %e\n", i, ii, iv, crealf(Rpm[i*n2+ii*n1+iv]), cimagf(Rpm[i*n2+ii*n1+iv]));
    //fprintf(stderr,"P=%e %e\n",  crealf((p1+p2)*tRmp/p3), cimagf((p1+p2))*tRmp/p3);
    fprintf(stderr,"P[%d][%d][%d]=%e %e\n", i, ii, iv, crealf(P[i*n2+ii*n1+iv]), cimagf(P[i*n2+ii*n1+iv]));
*/
                        }
                    }
                }
            }
        }
        else{           // rec above (up) / below (down) src layer
//fprintf(stderr,"rec above (up) / below (down) src layer up=%d lsrc=%d lrec=%d last_layer=%d first_layer=%d dp=%d nlsr=%d\n", up, lsrc, lrec, last_layer, first_layer, dp,nlsr);
            //           // Eqs  95/96,  A-24/A-25 for rec above src layer
            //           // Eqs 103/104, A-32/A-33 for rec below src layer
    
            // First compute P_{s-1} (up) / P_{s+1} (down)
            if (first_layer){  // If src is in bottom (up) / top (down) layer
                for (i=0;i<nfreq;i++) { 
                    for (ii=0;ii<noff;ii++) { 
#pragma ivdep
                        for (iv=0;iv<nlambda;iv++) { 
                            tiRpm = Rpm[i*m3+ii*m2+rsrcl*m1+iv];
                            tiGam = Gam[i*g3+ii*g2+lsrc*g1+iv];
                            P[i*n2+ii*n1+iv] = (1 + tiRpm)*mupm*cexp(-tiGam*dp);
                        }
                    }
                }
            }
            else{
                for (i=0;i<nfreq;i++) { 
                    for (ii=0;ii<noff;ii++) { 
#pragma ivdep
                        for (iv=0;iv<nlambda;iv++) { 
                            iRmp = Rmp[i*m3+ii*m2+rsrcl*m1+iv];
                            tRpm = Rpm[i*m3+ii*m2+rsrcl*m1+iv];
                            tiGam = Gam[i*g3+ii*g2+lsrc*g1+iv];
                            p1 = mupm*cexp(-tiGam*dp);
                            p2 = pm*mupm*iRmp*cexp(-tiGam * (ds+dm));
                            p3 = (1 + tRpm)/(1 - iRmp*tRpm*cexp(-2*tiGam*ds));
                            P[i*n2+ii*n1+iv] = (p1 + p2) * p3;
                        }
                    }
                }
            }

            // If up or down and src is in last but one layer
            if (up==1 || (up==0 && ((lsrc+1) < nlayer-1))) {
                ddepth = depth[lsrc+1-1*pup]-depth[lsrc-1*pup];
                if (!isinf(ddepth)) {
                    for (i=0;i<nfreq;i++) { 
                        for (ii=0;ii<noff;ii++) { 
                            #pragma ivdep
                            for (iv=0;iv<nlambda;iv++) { 
                                tiRpm = Rpm[i*m3+ii*m2+(rsrcl-1*pup)*m1+iv];
                                P[i*n2+ii*n1+iv] = P[i*n2+ii*n1+iv] / (1 + tiRpm*cexp(-2*Gam[i*g3+ii*g2+(lsrc-1*pup)*g1+iv]*ddepth));
                            }
                        }
                    }
                }
            }

            // Second compute P for all other layers
            if (nlsr > 2){
                for (iz=izstart;iz<izend;iz++) { 
                    ddepth = depth[isr+iz+pup+1]-depth[isr+iz+pup];
                    for (i=0;i<nfreq;i++) { 
                        for (ii=0;ii<noff;ii++) { 
#pragma ivdep
                            for (iv=0;iv<nlambda;iv++) { 
                                tiRpm = Rpm[i*m3+ii*m2+(iz+pup)*m1+iv];
                                piGam = Gam[i*g3+ii*g2+(isr+iz+pup)*g1+iv];
                                p1 = (1+tiRpm)*cexp(-piGam*ddepth);
                                P[i*n2+ii*n1+iv] *= p1;
                            }
                        }
                    }

                    // If rec/src NOT in first/last layer (up/down)
                    if ((isr+iz) != last){
                        ddepth = depth[isr+iz+1] - depth[isr+iz];
                        for (i=0;i<nfreq;i++) { 
                            for (ii=0;ii<noff;ii++) { 
#pragma ivdep
                                for (iv=0;iv<nlambda;iv++) { 
                                    tiRpm = Rpm[i*m3+ii*m2+iz*m1+iv];
                                    piGam2 = Gam[i*g3+ii*g2+(isr+iz)*g1+iv];
                                    p1 = 1 + tiRpm*cexp(-2*piGam2 * ddepth);
                                    P[i*n2+ii*n1+iv] /= p1;
                                }
                            }
                        }
                    }
                }
            }
        }
    } // up 0,1 loop

    // Return fields (up- and downgoing)

    return;
}

