#include "check.h"
#include "c3tensor.h"


#define NR_END 1
#define FREE_ARG char*


fftw_complex ***c3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate an fftw_complex 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
        long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
        fftw_complex ***t;

        /* allocate pointers to pointers to rows */
        t=(fftw_complex ***) malloc((size_t)((nrow+NR_END)*sizeof(fftw_complex**)));
        check(t != NULL);
        t += NR_END;
        t -= nrl;

        /* allocate pointers to rows and set pointers to them */
        t[nrl]=(fftw_complex **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(fftw_complex*)));
        check(t[nrl] != NULL);
        t[nrl] += NR_END;
        t[nrl] -= ncl;

        /* allocate rows and set pointers to them */
        t[nrl][ncl]=(fftw_complex *) fftw_malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(fftw_complex)));
        check(t[nrl][ncl] != NULL);
        t[nrl][ncl] += NR_END;
        t[nrl][ncl] -= ndl;

        for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
        for(i=nrl+1;i<=nrh;i++) {
                t[i]=t[i-1]+ncol;
                t[i][ncl]=t[i-1][ncl]+ncol*ndep;
                for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
        }

        /* return pointer to array of pointers to rows */
        return t;
}


void free_c3tensor(fftw_complex ***t, long nrl, long nrh, long ncl, long nch,
        long ndl, long ndh)
/* free an fftw_complex c3tensor allocated by c3tensor() */
{
        unused(nrh); unused(nch); unused(ndh);

        fftw_free((FREE_ARG) (t[nrl][ncl]+ndl-NR_END));
        free((FREE_ARG) (t[nrl]+ncl-NR_END));
        free((FREE_ARG) (t+nrl-NR_END));
}
