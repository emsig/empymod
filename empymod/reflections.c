#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<stdbool.h>
#include<complex.h>

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

void reflections(int nfreq, int noff, int nlayer, int nlambda, double *depth, double complex *e_zH, double complex *Gam, int lrec, int lsrc, double complex *Rp, double complex *Rm) 
{
    int maxl, minl, pm, izout, minmax, izout0;
    int *layer_count, lcount;
    bool shiftplus, shiftminus;
    int g1, g2, g3, n1, n2, n3, nlay;
    int i, ii, iv, iz;
    double complex rloc, *tRef, *out, *Ref;
    double complex ra, rb, rloca, rlocb, term;
    double ddepth;

    g1 = nlambda;
    g2 = g1*nlayer;
    g3 = g2*noff;

    // Get numbers and max/min layer.
    maxl = MAX(lrec, lsrc);
    minl = MIN(lrec, lsrc);
    nlay = (maxl-minl+1);
    n1 = nlambda;
    n2 = n1*nlay;
    n3 = n2*noff;

    //fprintf(stderr,"nfreq=%d noff=%d nlayer=%d nlambda=%d lrec=%d lsrc=%d nlay=%d\n",nfreq, noff, nlayer, nlambda, lrec, lsrc,nlay);
    //fprintf(stderr,"r2=%d r1=%d n3=%d n2=%d n1=%d\n",r2, r1, n3, n2, n1);
    // Pre-allocate tRef 
    // TODO eliminate tRef array
    tRef = (double complex *)malloc(nlambda*sizeof(double complex));
    layer_count = (int *)malloc(sizeof(int) * nlayer);

    // Loop over Rp, Rm
    for (int plus = 0; plus <= 1; plus++) {

        // Switches depending if plus or minus
        if (plus==1) {
            pm = 1;
            //layer_count = np.arange(depth.size-2, minl-1, -1)
            for (int i = nlayer - 2; i > minl-1; i--) {
                layer_count[nlayer-2 - i] = i;
//                fprintf(stderr,"plus=%d i=%d layer_count[%d] = %d \n",plus, i, nlayer - 2 - i, layer_count[nlayer - 2 - i]);
            }
            lcount = nlayer-2-(minl-1);
            izout = abs(lsrc-lrec);
            minmax = pm*maxl;
            out = Rp;
        }
        else {
            pm = -1;
            //layer_count = np.arange(1, maxl+1, 1)
            for (int i = 0; i < maxl; i++) {
                layer_count[i] = i + 1;
//                fprintf(stderr,"plus=%d i=%d layer_count[%d] = %d \n",plus, i, i, layer_count[i]);
            }
            lcount = maxl;
            izout = 0;
            minmax = pm*minl;
            out = Rm;
        }

        // If rec in last  and rec below src (plus) or
        // if rec in first and rec above src (minus), shift izout
        if ( (lrec<lsrc) && (lrec==0) && (plus==0)) {shiftplus = true;}
        else {shiftplus = false;}
        if ( (lrec>lsrc) && (lrec==nlayer-1) && (plus==1)) {shiftminus = true;}
        else {shiftminus = false;}
        if (shiftplus || shiftminus) {izout -= pm;}

        izout0=izout;

        // Calculate the reflection
        // Eqs 65, A-12
        for (i=0;i<nfreq;i++) {
            for (ii=0;ii<noff;ii++) {

                izout = izout0;
                for (int il=0; il<lcount; il++) {
                    iz = layer_count[il];
                    ra = e_zH[i*nlayer+iz+pm];
                    rb = e_zH[i*nlayer+iz];
                    // In first layer tRef = rloc
                    if (iz == layer_count[0]) {
                        for (iv=0;iv<nlambda;iv++) {
                            rloca = ra*Gam[i*g3+ii*g2+iz*g1+iv];
                            rlocb = rb*Gam[i*g3+ii*g2+(iz+pm)*g1+iv];
                            tRef[iv] = (rloca - rlocb)/(rloca + rlocb);
                        }
                    }
                    else{
                        ddepth = depth[iz+1+pm]-depth[iz+pm];;
    
                        // Eqs 64, A-11
                        for (iv=0;iv<nlambda;iv++) {
                            term = tRef[iv]*cexp(
                                    -2*Gam[i*g3+ii*g2+(iz+pm)*g1+iv]*ddepth);
                            rloca = ra*Gam[i*g3+ii*g2+iz*g1+iv];
                            rlocb = rb*Gam[i*g3+ii*g2+(iz+pm)*g1+iv];
                            rloc = (rloca - rlocb)/(rloca + rlocb);
                            tRef[iv] = (rloc + term)/(1 + rloc*term);
                        }
                    }

                    // The global reflection coefficient is given back for all layers
                    // between and including src- and rec-layer
                    if ((lrec != lsrc) && (pm*iz <= minmax)) {
                        for (int iv = 0; iv < nlambda; iv++) {
                            out[i*n3 + ii*n2 + izout*n1 + iv] = tRef[iv];
                        }
                    }
    
                    // If lsrc = lrec, we just store the last values
                    if (lsrc == lrec && lcount > 0 && il == lcount-1 ) {
                        for (int iv = 0; iv < nlambda; iv++) {
                            out[i*n3 + ii*n2 + iv] = tRef[iv];
                        }
                    }
                
                    if ((lrec != lsrc) && (pm*iz <= minmax)) {
                        izout -= pm;
                    }

                } // end of lcount layer loop
            } // end of noff loop
        }  // end of nfreq loop
        
    } // end of plus loop

    free(layer_count);
    free(tRef);

    // Return reflections (minus and plus)
    return;
}

