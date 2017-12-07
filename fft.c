#include "check.h"
#include "d3tensor.h"
#include "c3tensor.h"
#include "fft.h"


fftx_plan *create_fftx_plan(int nx,
    int xst_d[3], int xen_d[3], int xsz_d[3],
    int xst_c[3], int xen_c[3], int xsz_c[3])
{
    fftx_plan *p = (fftx_plan *) malloc(sizeof(fftx_plan));
    check(p != NULL);

    p->nx = nx;

    {
        int i;

        for (i = 0; i < 3; i++) {
            p->xst_d[i] = xst_d[i];
            p->xen_d[i] = xen_d[i];
            p->xsz_d[i] = xsz_d[i];
            p->xst_c[i] = xst_c[i];
            p->xen_c[i] = xen_c[i];
            p->xsz_c[i] = xsz_c[i];
        }
    }

    p->work_d = d3tensor(
        p->xst_d[0], p->xen_d[0],
        p->xst_d[1], p->xen_d[1],
        p->xst_d[2], p->xen_d[2]);
    p->work_c = c3tensor(
        p->xst_c[0], p->xen_c[0],
        p->xst_c[1], p->xen_c[1],
        p->xst_c[2], p->xen_c[2]);

    p->d2c = fftw_plan_many_dft_r2c(1, // rank
        (int []) {p->nx}, // n
        p->xsz_d[1] * p->xsz_d[2], // howmany
        &(p->work_d[p->xst_d[0]][p->xst_d[1]][p->xst_d[2]]), // in
        NULL, // inembed
        p->xsz_d[1] * p->xsz_d[2], // istride
        1, // idist
        &(p->work_c[p->xst_c[0]][p->xst_c[1]][p->xst_c[2]]), // out
        NULL, // onembed
        p->xsz_c[1] * p->xsz_c[2], // ostride
        1, // odist
        FFTW_MEASURE // flags
        );

    p->c2d = fftw_plan_many_dft_c2r(1, // rank
        (int []) {p->nx}, // n
        p->xsz_c[1] * p->xsz_c[2], // howmany
        &(p->work_c[p->xst_c[0]][p->xst_c[1]][p->xst_c[2]]), // in
        NULL, // inembed
        p->xsz_c[1] * p->xsz_c[2], // istride
        1, // idist
        &(p->work_d[p->xst_d[0]][p->xst_d[1]][p->xst_d[2]]), // out
        NULL, // onembed
        p->xsz_d[1] * p->xsz_d[2], // ostride
        1, // odist
        FFTW_MEASURE // flags
        );

    return p;
}


ffty_plan *create_ffty_plan(int ny,
    int yst_c[3], int yen_c[3], int ysz_c[3])
{
    ffty_plan *p = (ffty_plan *) malloc(sizeof(ffty_plan));
    check(p != NULL);

    p->ny = ny;

    {
        int i;

        for (i = 0; i < 3; i++) {
           p->yst_c[i] = yst_c[i];
           p->yen_c[i] = yen_c[i];
           p->ysz_c[i] = ysz_c[i];
        }
    }

    p->work_i = c3tensor(
        p->yst_c[0], p->yen_c[0],
        p->yst_c[1], p->yen_c[1],
        p->yst_c[2], p->yen_c[2]);
    p->work_o = c3tensor(
        p->yst_c[0], p->yen_c[0],
        p->yst_c[1], p->yen_c[1],
        p->yst_c[2], p->yen_c[2]);

    p->i2o = fftw_plan_guru_dft(1, // rank
        (fftw_iodim[]) {{p->ny, p->ysz_c[2], p->ysz_c[2]}}, // dims
        2, // howmany_rank
        (fftw_iodim[]) {
            {p->ysz_c[0], p->ysz_c[1] * p->ysz_c[2], p->ysz_c[1] * p->ysz_c[2]},
            {p->ysz_c[2], 1, 1}}, // homany_dims
        &(p->work_i[p->yst_c[0]][p->yst_c[1]][p->yst_c[2]]), // in
        &(p->work_o[p->yst_c[0]][p->yst_c[1]][p->yst_c[2]]), // out
        FFTW_FORWARD, // sign
        FFTW_MEASURE // flags
        );

    p->o2i = fftw_plan_guru_dft(1, // rank
        (fftw_iodim[]) {{p->ny, p->ysz_c[2], p->ysz_c[2]}}, // dims
        2, // howmany_rank
        (fftw_iodim[]) {
            {p->ysz_c[0], p->ysz_c[1] * p->ysz_c[2], p->ysz_c[1] * p->ysz_c[2]},
            {p->ysz_c[2], 1, 1}}, // howmany_dims
        &(p->work_o[p->yst_c[0]][p->yst_c[1]][p->yst_c[2]]), // in
        &(p->work_i[p->yst_c[0]][p->yst_c[1]][p->yst_c[2]]), // out
        FFTW_BACKWARD, // sign
        FFTW_MEASURE // flags
        );

    return p;
}


void destroy_fftx_plan(fftx_plan *p)
{
    free_d3tensor(p->work_d,
        p->xst_d[0], p->xen_d[0],
        p->xst_d[1], p->xen_d[1],
        p->xst_d[2], p->xen_d[2]);
    free_c3tensor(p->work_c,
        p->xst_c[0], p->xen_c[0],
        p->xst_c[1], p->xen_c[1],
        p->xst_c[2], p->xen_c[2]);

    fftw_destroy_plan(p->d2c);
    fftw_destroy_plan(p->c2d);

    free(p);
}


void destroy_ffty_plan(ffty_plan *p)
{
    free_c3tensor(p->work_i,
        p->yst_c[0], p->yen_c[0],
        p->yst_c[1], p->yen_c[1],
        p->yst_c[2], p->yen_c[2]);
    free_c3tensor(p->work_o,
        p->yst_c[0], p->yen_c[0],
        p->yst_c[1], p->yen_c[1],
        p->yst_c[2], p->yen_c[2]);

    fftw_destroy_plan(p->i2o);
    fftw_destroy_plan(p->o2i);

    free(p);
}


void fftx_d2c(fftx_plan *p)
{
    fftw_execute(p->d2c);

    int i, j, k;

    double fft_scale = 1.0 / (double) p->nx;

    for (i = p->xst_c[0]; i <= p->xen_c[0]; i++) {
        for (j = p->xst_c[1]; j <= p->xen_c[1]; j++) {
            for (k = p->xst_c[2]; k <= p->xen_c[2]; k++) {
                p->work_c[i][j][k][0] *= fft_scale;
                p->work_c[i][j][k][1] *= fft_scale;
            }
        }
    }
}


void fftx_c2d(fftx_plan *p)
{
    fftw_execute(p->c2d);
}


void ffty_i2o(ffty_plan *p)
{
    fftw_execute(p->i2o);

    int i, j, k;

    double fft_scale = 1.0 / (double) p->ny;

    for (i = p->yst_c[0]; i <= p->yen_c[0]; i++) {
        for (j = p->yst_c[1]; j <= p->yen_c[1]; j++) {
            for (k = p->yst_c[2]; k <= p->yen_c[2]; k++) {
                p->work_o[i][j][k][0] *= fft_scale;
                p->work_o[i][j][k][1] *= fft_scale;
            }
        }
    }
}


void ffty_o2i(ffty_plan *p)
{
    fftw_execute(p->o2i);
}
