#include "check.h"
#include "d3tensor.h"
#include "c3tensor.h"
#include "decomp.h"
#include "transpose.h"
#include "fft.h"
#include "ghostz.h"
#include "ghost.h"
#include "get.h"
#include "io.h"
#include "rk3.h"
#include "pressure.h"
#include "diffuseu.h"
#include "diffusew.h"
#include "advect.h"
#include "cases.h"


void case0(int p_row, int p_col,
           int nx, int ny, int nz,
           int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    pressure_initialize2(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dt = 1.0;
    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        //dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
        //             - cos(M_PI * (double) (iz + 1) / (double) nz));
        dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***r_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *r = &(r_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    for (i = 0; i < local_size; i++) u[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) r[i] = FILL_VALUE;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++)
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = z[k] * z[k] * z[k] * z[k];
            }

    pressure_laplacian2(u, r, local_nx, local_ny, nz,
                        nghost_x, nghost_y, nghost_z, dx, dy, dz, dt);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (k = 0; k < n2; k++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                double exact = 12.0 * z[k] * z[k];
                double err = (r[ijk] - exact);
                // 2nd-order can differentiate z^3 exactly but not z^4
                if (ix == 0 && iy == 0)
                    printf("k=%d z=%+e dz=%+e u=%+e err=%+e "
                        "exact=%+e r=%+e\n",
                        k, z[k], dz[k], u[ijk], err, exact, r[ijk]);
            }

    if (nrank == 0) printf("\n");

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ijk = (i * n1 + j) * n2 + k;
                //r[ijk] = z[k] - 0.5;
                r[ijk] = - M_PI * M_PI * cos(M_PI * z[k]);
            }

    ghostz_periodic(r, ghostz_p);

    ghostz_truncate_double(r_3d, zwork_d, ghostz_p);
    transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
        transp_d);
    fftx_d2c(fftx_p);
    transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_i2o(ffty_p);
    transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        transp_c);

    pressure_poisson2(
        &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
        ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
        nz, nghost_z,
        dx, dy, dz, dt);

    transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_o2i(ffty_p);
    transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        transp_c);
    fftx_c2d(fftx_p);
    transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        transp_d);
    ghostz_pad_double(u_3d, zwork_d, ghostz_p);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (k = 0; k < n2; k++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                //double z2 = z[k] * z[k];
                //double z3 = z2 * z[k];
                //double exact = z3 / 6.0 - z2 / 4.0;
                double exact = cos(M_PI * z[k]) - 1.0;
                double err = (u[ijk] - exact) / exact;
                if (ix == 0 && iy == 0)
                    printf("k=%d z=%+e dz=%+e u=%+e err=%+e "
                        "exact=%+e r=%+e\n",
                        k, z[k], dz[k], u[ijk], err, exact, r[ijk]);
            }

    free(dz);
    free(z);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(r_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);

    pressure_finalize2();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


void case1(int p_row, int p_col,
           int nx, int ny, int nz,
           int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    if (nrank == 0)
        printf("--------\nCheck 2nd-order pressure Poisson solver\n--------\n");

    pressure_initialize2(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dt = 1.0;
    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        //dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
        //             - cos(M_PI * (double) (iz + 1) / (double) nz));
        dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***v_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***w_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***p_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***r_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***d_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***l_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *v = &(v_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *w = &(w_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *p = &(p_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *r = &(r_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *d = &(d_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *l = &(l_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    srand(nrank);

    for (i = 0; i < local_size; i++) u[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) v[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) w[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) p[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) r[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) d[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) l[i] = FILL_VALUE;

    ghostz_periodic(u, ghostz_p);
    ghostz_periodic(v, ghostz_p);
    ghostz_periodic(w, ghostz_p);

    double avg_bot, avg_top;
    get_avg_blowing_z(w, local_nx, local_ny, nz,
                      nghost_x, nghost_y, nghost_z,
                      nx, ny, &avg_bot, &avg_top);
    if (nrank == 0)
        printf("avg_bot=%+e avg_top=%+e\n", avg_bot, avg_top);
    divergence2(u, v, w, r, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, dx, dy, dz);

    ghostz_truncate_double(r_3d, zwork_d, ghostz_p);
    transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
        transp_d);
    fftx_d2c(fftx_p);
    transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_i2o(ffty_p);
    transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        transp_c);

    pressure_poisson2(
        &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
        ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
        nz, nghost_z,
        dx, dy, dz, dt);

    transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_o2i(ffty_p);
    transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        transp_c);
    fftx_c2d(fftx_p);
    transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        transp_d);
    ghostz_pad_double(p_3d, zwork_d, ghostz_p);

    ghostz_periodic(p, ghostz_p);
    project2(u, v, w, p, local_nx, local_ny, nz,
             nghost_x, nghost_y, nghost_z, dx, dy, dz, dt);

    divergence2(u, v, w, d, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, dx, dy, dz);
    pressure_laplacian2(p, l, local_nx, local_ny, nz,
                        nghost_x, nghost_y, nghost_z, dx, dy, dz, dt);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                if (fabs(d[ijk]) > 1e-10) // Check zero divergence
                    printf("ix=%d iy=%d iz=%d div=%+e\n",
                        ix, iy, iz, d[ijk]);
            }

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                double err = r[ijk] - l[ijk];
                if (fabs(err) > 1e-10) // Check pressure Poisson equation
                    printf("ix=%d iy=%d iz=%d err=%+e\n",
                        ix, iy, iz, err);
            }

    get_avg_blowing_z(w, local_nx, local_ny, nz,
                      nghost_x, nghost_y, nghost_z,
                      nx, ny, &avg_bot, &avg_top);
    if (nrank == 0)
        printf("avg_bot=%+e avg_top=%+e\n", avg_bot, avg_top);

    free(dz);
    free(z);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(v_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(w_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(p_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(r_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(d_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(l_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);

    pressure_finalize2();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


static double bc_bot_case2(double x, double y) { return x * y + 1.0; }
static double bc_top_case2(double x, double y) { return x + y - 3.0; }

void case2(int p_row, int p_col,
           int nx, int ny, int nz,
           int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    diffuseu_initialize2(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *xu = (double *) malloc(sizeof(double) * n0);
    double *yp = (double *) malloc(sizeof(double) * n1);
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(xu != NULL);
    check(yp != NULL);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (i = 0; i < n0; i++) {
        local_ix = i - nghost_x;
        int ix = decomp_d->zst[0] - 1 + local_ix;
        ix = (ix + nx) % nx; // ix = 0, 1, ..., nx - 1
        xu[i] = (double) (ix + 1) * dx;
    }

    for (j = 0; j < n1; j++) {
        local_iy = j - nghost_y;
        int iy = decomp_d->zst[1] - 1 + local_iy;
        iy = (iy + ny) % ny; // iy = 0, 1, ..., ny - 1
        yp[j] = ((double) (iy + 1) - 0.5) * dy;
    }

    for (k = 0; k < n2; k++) dz[k] = FILL_VALUE;
    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
                     - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = 1.0 / (double) nz;
    }
    //ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double   ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double ***ubc_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double  ***u1_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double   ***h_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double ***hbc_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);

    double   *u =   &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *ubc = &(ubc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double  *u1 =  &(u1_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double   *h =   &(h_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *hbc = &(hbc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    srand(nrank);

    for (i = 0; i < local_size; i++) u[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) u1[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) h[i] = FILL_VALUE;

    ghostz_periodic(u, ghostz_p);

    double bctype = NEUMANU;
    int test1d = 0;
    if (test1d == 1) {
        if (nrank == 0) printf("----Testing 1D helmholtz solver----\n");
    }
    else {
        if (nrank == 0) printf("----Testing 3D helmholtz solver----\n");
    }

    double nudt = 2.0;
    apply_bc_u2(u, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xu, yp, dz, bc_bot_case2, bc_top_case2, bctype, bctype);
    if (test1d == 1) {
        helmholtz1d_operator_u2(u, h, local_nx, local_ny, nz,
                                nghost_x, nghost_y, nghost_z, dz, nudt);
    }
    else {
        helmholtz_operator_u2(u, h, local_nx, local_ny, nz,
                              nghost_x, nghost_y, nghost_z, dx, dy, dz, nudt);
    }
    //Done setting h here.

    for (i = 0; i < local_size; i++) ubc[i] = 0.0;
    apply_bc_u2(ubc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xu, yp, dz, bc_bot_case2, bc_top_case2, bctype, bctype);
    if (test1d == 1) {
        helmholtz1d_operator_u2(ubc, hbc, local_nx, local_ny, nz,
                                nghost_x, nghost_y, nghost_z, dz, nudt);
    }
    else {
        helmholtz_operator_u2(ubc, hbc, local_nx, local_ny, nz,
                              nghost_x, nghost_y, nghost_z, dx, dy, dz, nudt);
    }
    for (i = 0; i < local_size; i++) h[i] -= hbc[i];

    if (test1d == 1) {
        solve_helmholtz1d_u2(u1, h, local_nx, local_ny, nz,
                             nghost_x, nghost_y, nghost_z, dz, nudt,
                             0.0, 0.0, bctype, bctype);
    }
    else {
        ghostz_truncate_double(h_3d, zwork_d, ghostz_p);
        transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
            transp_d);
        fftx_d2c(fftx_p);
        transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
            transp_c);
        ffty_i2o(ffty_p);
        transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
            transp_c);

        solve_helmholtz_u2(
            &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
            nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
            ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
            nz, nghost_z,
            dx, dy, dz, nudt, 0.0, 0.0, bctype, bctype);

        transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
            transp_c);
        ffty_o2i(ffty_p);
        transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
            transp_c);
        fftx_c2d(fftx_p);
        transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
            transp_d);
        ghostz_pad_double(u1_3d, zwork_d, ghostz_p);
    }
    //Put bc back:
    apply_bc_u2(u1, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xu, yp, dz, bc_bot_case2, bc_top_case2, bctype, bctype);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = -1; iz < nz + 1; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                double err = u[ijk] - u1[ijk];
                if (fabs(err) > 1e-9) // Check forward-backward consistency
                                       // of Helmholtz operator/solver
                    printf("ix=%d iy=%d iz=%d u=%+e u1=%+e err=%+e\n",
                        ix, iy, iz, u[ijk], u1[ijk], err);
            }

    double *u_avg = (double *) malloc(sizeof(double) * n2);

    get_xy_avg(u1, u_avg, local_nx, local_ny, nz,
        nghost_x, nghost_y, nghost_z, nx, ny);
    if (nrank == 0)
        for (k = 0; k < n2; k++) printf("%+e\n", u_avg[k]);

    free(u_avg);

    free(xu);
    free(yp);
    free(dz);
    free(z);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d,   zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(ubc_3d, zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(u1_3d,  zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(h_3d,   zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(hbc_3d, zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);

    diffuseu_finalize2();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


void case3(int p_row, int p_col,
           int nx, int ny, int nz,
           int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    diffuseu_initialize2(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
                     - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***h_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *h = &(h_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    double *u_avg = (double *) malloc(sizeof(double) * n2);
    double *u_var = (double *) malloc(sizeof(double) * n2);
    check(u_avg != NULL);
    check(u_var != NULL);

    srand(nrank);

    for (i = 0; i < local_size; i++) {
        u[i] = 0.5 + (double) rand() / (double) RAND_MAX;
    }
    for (i = 0; i < local_size; i++) h[i] = FILL_VALUE;

    int step, it, nt = 5, istat = 5;
    double nu = 0.1, dt = 10.0;

    for (it = 0; it < nt; it++) {

        for (step = 0; step < 3; step++) { // Time marching of 2nd-order u-diffusion

            double nuadt = nu * rk3_alp[step] * dt;
            double nubdt = nu * rk3_bet[step] * dt;

            ghostz_periodic(u, ghostz_p);

            helmholtz_operator_u2(u, h, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt);

            ghostz_truncate_double(h_3d, zwork_d, ghostz_p);
            transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
                transp_d);
            fftx_d2c(fftx_p);
            transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
                transp_c);
            ffty_i2o(ffty_p);
            transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                transp_c);

            solve_helmholtz_u2(
            &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt, 0.0, 0.0, DIRICHU, DIRICHU);

            transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
                transp_c);
            ffty_o2i(ffty_p);
            transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
                transp_c);
            fftx_c2d(fftx_p);
            transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
                transp_d);
            ghostz_pad_double(u_3d, zwork_d, ghostz_p);
        }

        double t = dt * (double) (it + 1);

        for (local_ix = 0; local_ix < local_nx; local_ix++)
            for (local_iy = 0; local_iy < local_ny; local_iy++)
                for (iz = 0; iz < nz; iz++) {
                    i = local_ix + nghost_x;
                    j = local_iy + nghost_y;
                    k = iz + nghost_z;
                    int ix = local_ix + decomp_d->zst[0] - 1;
                    int iy = local_iy + decomp_d->zst[1] - 1;
                    int ijk = (i * n1 + j) * n2 + k;
                    if (ix == 0 && iy == 0 && iz == nz / 2)
                        printf("%+e %+e %+e\n", t, z[k], u[ijk]);
                }

        if ((it + 1) % istat == 0) {
            get_xy_avg(u, u_avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            get_xy_corr(u, u, u_var,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            if (nrank == 0)
                for (k = 0; k < n2; k++)
                    printf("%+e %+e %+e %+e\n", t, z[k], u_avg[k], u_var[k]);
        }
    }

    free(dz);
    free(z);
    free(u_avg);
    free(u_var);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(h_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);

    diffuseu_finalize2();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


static double bc_bot_case4(double x, double y) { return x * y - 2.0; }
static double bc_top_case4(double x, double y) { return x + y + 1.0; }

void case4(int p_row, int p_col,
           int nx, int ny, int nz,
           int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    diffusew_initialize2(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *xp = (double *) malloc(sizeof(double) * n0);
    double *yp = (double *) malloc(sizeof(double) * n1);
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(xp != NULL);
    check(yp != NULL);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (i = 0; i < n0; i++) {
        local_ix = i - nghost_x;
        int ix = decomp_d->zst[0] - 1 + local_ix;
        ix = (ix + nx) % nx; // ix = 0, 1, ..., nx - 1
        xp[i] = ((double) (ix + 1) - 0.5) * dx;
    }

    for (j = 0; j < n1; j++) {
        local_iy = j - nghost_y;
        int iy = decomp_d->zst[1] - 1 + local_iy;
        iy = (iy + ny) % ny; // iy = 0, 1, ..., ny - 1
        yp[j] = ((double) (iy + 1) - 0.5) * dy;
    }

    for (k = 0; k < n2; k++) dz[k] = FILL_VALUE;
    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
                     - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = 1.0 / (double) nz;
    }
    //ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double   ***w_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double ***wbc_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double  ***w1_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double   ***h_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double ***hbc_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);

    double   *w =   &(w_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *wbc = &(wbc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double  *w1 =  &(w1_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double   *h =   &(h_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *hbc = &(hbc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    srand(nrank);

    for (i = 0; i < local_size; i++) w[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) w1[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) h[i] = FILL_VALUE;

    ghostz_periodic(w, ghostz_p);

    double bctype = DIRICHW;
    int test1d = 1;
    if (test1d == 1) {
        if (nrank == 0) printf("----Testing 1D helmholtz solver----\n");
    }
    else {
        if (nrank == 0) printf("----Testing 3D helmholtz solver----\n");
    }

    double nudt = 0.7;
    apply_bc_w2(w, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xp, yp, dz, bc_bot_case4, bc_top_case4, bctype, bctype);
    if (test1d == 1) {
        helmholtz1d_operator_w2(w, h, local_nx, local_ny, nz,
                                nghost_x, nghost_y, nghost_z, dz, nudt);
    }
    else {
        helmholtz_operator_w2(w, h, local_nx, local_ny, nz,
                              nghost_x, nghost_y, nghost_z, dx, dy, dz, nudt);
    }
    //Done setting h here.

    for (i = 0; i < local_size; i++) wbc[i] = 0.0;
    apply_bc_w2(wbc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xp, yp, dz, bc_bot_case4, bc_top_case4, bctype, bctype);
    if (test1d == 1) {
        helmholtz1d_operator_w2(wbc, hbc, local_nx, local_ny, nz,
                                nghost_x, nghost_y, nghost_z, dz, nudt);
    }
    else {
        helmholtz_operator_w2(wbc, hbc, local_nx, local_ny, nz,
                              nghost_x, nghost_y, nghost_z, dx, dy, dz, nudt);
    }
    for (i = 0; i < local_size; i++) h[i] -= hbc[i];

    if (test1d == 1) {
        solve_helmholtz1d_w2(w1, h, local_nx, local_ny, nz,
                             nghost_x, nghost_y, nghost_z, dz, nudt,
                             0.0, 0.0, bctype, bctype);
    }
    else {
        ghostz_truncate_double(h_3d, zwork_d, ghostz_p);
        transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
            transp_d);
        fftx_d2c(fftx_p);
        transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
            transp_c);
        ffty_i2o(ffty_p);
        transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
            transp_c);

        solve_helmholtz_w2(
            &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
            nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
            ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
            nz, nghost_z,
            dx, dy, dz, nudt, 0.0, 0.0, bctype, bctype);

        transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
            transp_c);
        ffty_o2i(ffty_p);
        transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
            transp_c);
        fftx_c2d(fftx_p);
        transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
            transp_d);
        ghostz_pad_double(w1_3d, zwork_d, ghostz_p);
    }
    //Put bc back:
    apply_bc_w2(w1, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xp, yp, dz, bc_bot_case4, bc_top_case4, bctype, bctype);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = -1; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                double err = w[ijk] - w1[ijk];
                if (fabs(err) > 1e-9) // Check forward-backward consistency
                                       // of Helmholtz operator/solver
                    printf("ix=%d iy=%d iz=%d w=%+e w1=%+e err=%+e\n",
                        ix, iy, iz, w[ijk], w1[ijk], err);
            }

    double *w_avg = (double *) malloc(sizeof(double) * n2);

    get_xy_avg(w1, w_avg, local_nx, local_ny, nz,
        nghost_x, nghost_y, nghost_z, nx, ny);
    if (nrank == 0)
        for (k = 0; k < n2; k++) printf("%+e\n", w_avg[k]);

    free(w_avg);

    free(xp);
    free(yp);
    free(dz);
    free(z);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(w_3d,   zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(wbc_3d, zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(w1_3d,  zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(h_3d,   zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(hbc_3d, zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);

    diffusew_finalize2();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


void case5(int p_row, int p_col,
           int nx, int ny, int nz,
           int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    diffusew_initialize2(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
                     - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_w(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***w_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***h_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);

    double *w = &(w_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *h = &(h_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    double *w_avg = (double *) malloc(sizeof(double) * n2);
    double *w_var = (double *) malloc(sizeof(double) * n2);
    check(w_avg != NULL);
    check(w_var != NULL);

    srand(nrank);

    for (i = 0; i < local_size; i++) {
        w[i] = 0.5 + (double) rand() / (double) RAND_MAX;
    }
    for (i = 0; i < local_size; i++) h[i] = FILL_VALUE;

    int step, it, nt = 5, istat = 5;
    double nu = 0.1, dt = 10.0;

    for (it = 0; it < nt; it++) {

        for (step = 0; step < 3; step++) { // Time marching of 2nd-order w-diffusion

            double nuadt = nu * rk3_alp[step] * dt;
            double nubdt = nu * rk3_bet[step] * dt;

            ghostz_periodic(w, ghostz_p);

            helmholtz_operator_w2(w, h, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt);

            ghostz_truncate_double(h_3d, zwork_d, ghostz_p);
            transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
                transp_d);
            fftx_d2c(fftx_p);
            transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
                transp_c);
            ffty_i2o(ffty_p);
            transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                transp_c);

            solve_helmholtz_w2(
            &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt, 0.0, 0.0, DIRICHW, DIRICHW);

            transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
                transp_c);
            ffty_o2i(ffty_p);
            transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
                transp_c);
            fftx_c2d(fftx_p);
            transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
                transp_d);
            ghostz_pad_double(w_3d, zwork_d, ghostz_p);
        }

        double t = dt * (double) (it + 1);

        for (local_ix = 0; local_ix < local_nx; local_ix++)
            for (local_iy = 0; local_iy < local_ny; local_iy++)
                for (iz = 0; iz < nz; iz++) {
                    i = local_ix + nghost_x;
                    j = local_iy + nghost_y;
                    k = iz + nghost_z;
                    int ix = local_ix + decomp_d->zst[0] - 1;
                    int iy = local_iy + decomp_d->zst[1] - 1;
                    int ijk = (i * n1 + j) * n2 + k;
                    if (ix == 0 && iy == 0 && iz == nz / 2)
                        printf("%+e %+e %+e\n", t, z[k], w[ijk]);
                }

        if ((it + 1) % istat == 0) {
            get_xy_avg(w, w_avg, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, nx, ny);
            get_xy_corr(w, w, w_var, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, nx, ny);
            if (nrank == 0)
                for (k = 0; k < n2; k++)
                    printf("%+e %+e %+e %+e\n", t, z[k], w_avg[k], w_var[k]);
        }
    }

    free(dz);
    free(z);
    free(w_avg);
    free(w_var);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(w_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(h_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);

    diffusew_finalize2();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


static double bc_u_bot_case6(double x, double y) { unused(x); unused(y); return 10.0; }
static double bc_u_top_case6(double x, double y) { unused(x); unused(y); return 0.0; }
static double bc_v_bot_case6(double x, double y) { unused(x); unused(y); return 10.0; }
static double bc_v_top_case6(double x, double y) { unused(x); unused(y); return 0.0; }
static double bc_w_bot_case6(double x, double y) { unused(x); unused(y); return 3.0; }
static double bc_w_top_case6(double x, double y) { unused(x); unused(y); return 3.0; }

void case6(int p_row, int p_col,
           int nx, int ny, int nz,
           int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    if (nrank == 0)
        printf("--------\nCheck 2nd-order advective stress conservation\n--------\n");

    pressure_initialize2(nz, nghost_z);
    diffuseu_initialize2(nz, nghost_z);
    diffusew_initialize2(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *xp = (double *) malloc(sizeof(double) * n0);
    double *xu = (double *) malloc(sizeof(double) * n0);
    double *yp = (double *) malloc(sizeof(double) * n1);
    double *yv = (double *) malloc(sizeof(double) * n1);
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *zp = (double *) malloc(sizeof(double) * n2);
    check(xp != NULL);
    check(xu != NULL);
    check(yp != NULL);
    check(yv != NULL);
    check(dz != NULL);
    check(zp != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (i = 0; i < n0; i++) {
        local_ix = i - nghost_x;
        int ix = decomp_d->zst[0] - 1 + local_ix;
        ix = (ix + nx) % nx; // ix = 0, 1, ..., nx - 1
        xu[i] = (double) (ix + 1) * dx;
        xp[i] = ((double) (ix + 1) - 0.5) * dx;
    }

    for (j = 0; j < n1; j++) {
        local_iy = j - nghost_y;
        int iy = decomp_d->zst[1] - 1 + local_iy;
        iy = (iy + ny) % ny; // iy = 0, 1, ..., ny - 1
        yv[j] = (double) (iy + 1) * dy;
        yp[j] = ((double) (iy + 1) - 0.5) * dy;
    }

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 1.0 * 0.5 * (cos(M_PI * (double) iz / (double) nz)
                          - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, zp, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***v_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***w_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***d_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***p_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***muphx_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mvphx_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mwphx_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***muphy_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mvphy_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mwphy_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***muphz_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mvphz_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mwphz_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***fu_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***fv_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***fw_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *v = &(v_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *w = &(w_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *d = &(d_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *p = &(p_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *muphx = &(muphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mvphx = &(mvphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mwphx = &(mwphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *muphy = &(muphy_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mvphy = &(mvphy_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mwphy = &(mwphy_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *muphz = &(muphz_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mvphz = &(mvphz_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mwphz = &(mwphz_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fu = &(fu_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fv = &(fv_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fw = &(fw_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    srand(nrank);
 
    for (i = 0; i < local_size; i++) u[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) v[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) w[i] = (double) rand() / (double) RAND_MAX;

    ghostz_periodic(u, ghostz_p);
    apply_bc_u2(u, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xu, yp, dz, bc_u_bot_case6, bc_u_top_case6, DIRICHU, DIRICHU);
    ghostz_periodic(v, ghostz_p);
    apply_bc_u2(v, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xp, yv, dz, bc_v_bot_case6, bc_v_top_case6, DIRICHU, DIRICHU);
    ghostz_periodic(w, ghostz_p);
    apply_bc_w2(w, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xp, yp, dz, bc_w_bot_case6, bc_w_top_case6, DIRICHW, DIRICHW);
    divergence2(u, v, w, d, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, dx, dy, dz);
    ghostz_truncate_double(d_3d, zwork_d, ghostz_p);
    transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
        transp_d);
    fftx_d2c(fftx_p);
    transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_i2o(ffty_p);
    transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        transp_c);

    pressure_poisson2(
        &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
        ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
        nz, nghost_z,
        dx, dy, dz, 1.0);

    transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_o2i(ffty_p);
    transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        transp_c);
    fftx_c2d(fftx_p);
    transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        transp_d);
    ghostz_pad_double(p_3d, zwork_d, ghostz_p);
    ghostz_periodic(p, ghostz_p);
    project2(u, v, w, p, local_nx, local_ny, nz,
             nghost_x, nghost_y, nghost_z, dx, dy, dz, 1.0);
    ghostz_periodic(u, ghostz_p);
    ghostz_periodic(v, ghostz_p);
    ghostz_periodic(w, ghostz_p);

    massfluxx2(u, v, w, muphx, mvphx, mwphx,
               local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);
    advectu2(u, muphx, mvphx, mwphx, fu,
             local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);

    massfluxy2(u, v, w, muphy, mvphy, mwphy,
               local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);
    advectv2(v, muphy, mvphy, mwphy, fv,
             local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);

    massfluxz2(u, v, w, muphz, mvphz, mwphz,
               local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);
    advectw2(w, muphz, mvphz, mwphz, fw,
             local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);

    double ksumu = 0.0;
    double ksumv = 0.0;
    double ksumw = 0.0;
    double msumu = 0.0;
    double msumv = 0.0;
    double msumw = 0.0;

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ijk = (i * n1 + j) * n2 + k;
                ksumu +=  u[ijk] * dx * dy * dz[k] * fu[ijk];
                ksumv +=  v[ijk] * dx * dy * dz[k] * fv[ijk];
                msumu +=           dx * dy * dz[k] * fu[ijk];
                msumv +=           dx * dy * dz[k] * fv[ijk];
            }

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz - 1; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ijk = (i * n1 + j) * n2 + k;
                ksumw +=  w[ijk] * dx * dy * 0.5 * (dz[k] + dz[k + 1]) * fw[ijk];
                msumw +=           dx * dy * 0.5 * (dz[k] + dz[k + 1]) * fw[ijk];
            }

    MPI_Allreduce(MPI_IN_PLACE, &ksumu, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ksumv, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ksumw, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &msumu, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &msumv, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &msumw, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (nrank == 0) {
        // Can check -(78) Sanderse (2014) J. Comput. Phys. 257:1472-1505.
        // -(76) cannot be checked because the boundary tendencies are never computed.
        printf("ksumu = %+e, ksumv = %+e, ksumw = %+e\n", ksumu, ksumv, ksumw);
        printf("msumu = %+e, msumv = %+e, msumw = %+e\n", msumu, msumv, msumw);
    }

    free(xp);
    free(xu);
    free(yp);
    free(yv);
    free(dz);
    free(zp);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(v_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(w_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(d_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(p_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(muphx_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mvphx_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mwphx_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(muphy_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mvphy_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mwphy_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(muphz_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mvphz_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mwphz_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(fu_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(fv_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(fw_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);

    pressure_finalize2();
    diffuseu_finalize2();
    diffusew_finalize2();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


void case7(int p_row, int p_col,
           int nx, int ny, int nz,
           int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    pressure_initialize4(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dt = 1.0;
    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        //dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
        //             - cos(M_PI * (double) (iz + 1) / (double) nz));
        dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***r_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *r = &(r_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    for (i = 0; i < local_size; i++) u[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) r[i] = FILL_VALUE;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++)
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = z[k] * z[k] * z[k] * z[k] * z[k] * z[k];
            }

    pressure_laplacian4(u, r, local_nx, local_ny, nz,
                        nghost_x, nghost_y, nghost_z, dx, dy, dz, dt);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (k = 0; k < n2; k++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                double exact = 30.0 * z[k] * z[k] * z[k] * z[k];
                double err = (r[ijk] - exact);
                // 4th-order can differentiate z^5 exactly but not z^6
                if (ix == 0 && iy == 0)
                    printf("k=%d z=%+e dz=%+e u=%+e err=%+e "
                        "exact=%+e r=%+e\n",
                        k, z[k], dz[k], u[ijk], err, exact, r[ijk]);
            }

    if (nrank == 0) printf("\n");

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ijk = (i * n1 + j) * n2 + k;
                //r[ijk] = z[k] - 0.5;
                r[ijk] = - M_PI * M_PI * cos(M_PI * z[k]);
            }

    ghostz_periodic(r, ghostz_p);

    ghostz_truncate_double(r_3d, zwork_d, ghostz_p);
    transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
        transp_d);
    fftx_d2c(fftx_p);
    transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_i2o(ffty_p);
    transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        transp_c);

    pressure_poisson4(
        &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
        ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
        nz, nghost_z,
        dx, dy, dz, dt);

    transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_o2i(ffty_p);
    transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        transp_c);
    fftx_c2d(fftx_p);
    transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        transp_d);
    ghostz_pad_double(u_3d, zwork_d, ghostz_p);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (k = 0; k < n2; k++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                //double z2 = z[k] * z[k];
                //double z3 = z2 * z[k];
                //double exact = z3 / 6.0 - z2 / 4.0;
                double exact = cos(M_PI * z[k]) - 1.0;
                double err = (u[ijk] - exact) / exact;
                if (ix == 0 && iy == 0)
                    printf("k=%d z=%+e dz=%+e u=%+e err=%+e "
                        "exact=%+e r=%+e\n",
                        k, z[k], dz[k], u[ijk], err, exact, r[ijk]);
            }

    free(dz);
    free(z);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(r_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);

    pressure_finalize4();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


void case8(int p_row, int p_col,
           int nx, int ny, int nz,
           int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    if (nrank == 0)
        printf("--------\nCheck 4th-order pressure Poisson solver\n--------\n");

    pressure_initialize4(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dt = 1.0;
    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        //dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
        //             - cos(M_PI * (double) (iz + 1) / (double) nz));
        dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***v_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***w_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***p_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***r_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***d_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***l_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *v = &(v_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *w = &(w_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *p = &(p_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *r = &(r_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *d = &(d_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *l = &(l_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    srand(nrank);

    for (i = 0; i < local_size; i++) u[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) v[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) w[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) p[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) r[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) d[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) l[i] = FILL_VALUE;

    ghostz_periodic(u, ghostz_p);
    ghostz_periodic(v, ghostz_p);
    ghostz_periodic(w, ghostz_p);

    double avg_bot, avg_top;
    get_avg_blowing_z(w, local_nx, local_ny, nz,
                      nghost_x, nghost_y, nghost_z,
                      nx, ny, &avg_bot, &avg_top);
    if (nrank == 0)
        printf("avg_bot=%+e avg_top=%+e\n", avg_bot, avg_top);
    divergence4(u, v, w, r, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, dx, dy, dz);

    ghostz_truncate_double(r_3d, zwork_d, ghostz_p);
    transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
        transp_d);
    fftx_d2c(fftx_p);
    transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_i2o(ffty_p);
    transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        transp_c);

    pressure_poisson4(
        &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
        ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
        nz, nghost_z,
        dx, dy, dz, dt);

    transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_o2i(ffty_p);
    transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        transp_c);
    fftx_c2d(fftx_p);
    transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        transp_d);
    ghostz_pad_double(p_3d, zwork_d, ghostz_p);

    ghostz_periodic(p, ghostz_p);
    project4(u, v, w, p, local_nx, local_ny, nz,
             nghost_x, nghost_y, nghost_z, dx, dy, dz, dt);

    divergence4(u, v, w, d, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, dx, dy, dz);
    pressure_laplacian4(p, l, local_nx, local_ny, nz,
                        nghost_x, nghost_y, nghost_z, dx, dy, dz, dt);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                if (fabs(d[ijk]) > 1e-10) // Check zero divergence
                    printf("ix=%d iy=%d iz=%d div=%+e\n",
                        ix, iy, iz, d[ijk]);
            }

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                double err = r[ijk] - l[ijk];
                if (fabs(err) > 1e-10) // Check pressure Poisson equation
                    printf("ix=%d iy=%d iz=%d err=%+e\n",
                        ix, iy, iz, err);
            }

    get_avg_blowing_z(w, local_nx, local_ny, nz,
                      nghost_x, nghost_y, nghost_z,
                      nx, ny, &avg_bot, &avg_top);
    if (nrank == 0)
        printf("avg_bot=%+e avg_top=%+e\n", avg_bot, avg_top);

    free(dz);
    free(z);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(v_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(w_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(p_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(r_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(d_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(l_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);

    pressure_finalize4();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


static double bc_bot_case9(double x, double y) { return x * y + 1.0; }
static double bc_top_case9(double x, double y) { return x + y - 3.0; }

void case9(int p_row, int p_col,
           int nx, int ny, int nz,
           int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    diffuseu_initialize4(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *xu = (double *) malloc(sizeof(double) * n0);
    double *yp = (double *) malloc(sizeof(double) * n1);
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(xu != NULL);
    check(yp != NULL);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (i = 0; i < n0; i++) {
        local_ix = i - nghost_x;
        int ix = decomp_d->zst[0] - 1 + local_ix;
        ix = (ix + nx) % nx; // ix = 0, 1, ..., nx - 1
        xu[i] = (double) (ix + 1) * dx;
    }

    for (j = 0; j < n1; j++) {
        local_iy = j - nghost_y;
        int iy = decomp_d->zst[1] - 1 + local_iy;
        iy = (iy + ny) % ny; // iy = 0, 1, ..., ny - 1
        yp[j] = ((double) (iy + 1) - 0.5) * dy;
    }

    for (k = 0; k < n2; k++) dz[k] = FILL_VALUE;
    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
                     - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = 1.0 / (double) nz;
    }
    //ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double   ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double ***ubc_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double  ***u1_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double   ***h_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double ***hbc_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);

    double   *u =   &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *ubc = &(ubc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double  *u1 =  &(u1_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double   *h =   &(h_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *hbc = &(hbc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    srand(nrank);

    for (i = 0; i < local_size; i++) u[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) u1[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) h[i] = FILL_VALUE;

    ghostz_periodic(u, ghostz_p);

    double bctype = NEUMANU;
    int test1d = 0;
    if (test1d == 1) {
        if (nrank == 0) printf("----Testing 1D helmholtz solver----\n");
    }
    else {
        if (nrank == 0) printf("----Testing 3D helmholtz solver----\n");
    }

    double nudt = 2.0;
    apply_bc_u4(u, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xu, yp, dz, bc_bot_case9, bc_top_case9, bctype, bctype);
    if (test1d == 1) {
        helmholtz1d_operator_u4(u, h, local_nx, local_ny, nz,
                                nghost_x, nghost_y, nghost_z, dz, nudt);
    }
    else {
        helmholtz_operator_u4(u, h, local_nx, local_ny, nz,
                              nghost_x, nghost_y, nghost_z, dx, dy, dz, nudt);
    }
    //Done setting h here.

    for (i = 0; i < local_size; i++) ubc[i] = 0.0;
    apply_bc_u4(ubc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xu, yp, dz, bc_bot_case9, bc_top_case9, bctype, bctype);
    if (test1d == 1) {
        helmholtz1d_operator_u4(ubc, hbc, local_nx, local_ny, nz,
                                nghost_x, nghost_y, nghost_z, dz, nudt);
    }
    else {
        helmholtz_operator_u4(ubc, hbc, local_nx, local_ny, nz,
                              nghost_x, nghost_y, nghost_z, dx, dy, dz, nudt);
    }
    for (i = 0; i < local_size; i++) h[i] -= hbc[i];

    if (test1d == 1) {
        solve_helmholtz1d_u4(u1, h, local_nx, local_ny, nz,
                             nghost_x, nghost_y, nghost_z, dz, nudt,
                             0.0, 0.0, bctype, bctype);
    }
    else {
        ghostz_truncate_double(h_3d, zwork_d, ghostz_p);
        transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
            transp_d);
        fftx_d2c(fftx_p);
        transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
            transp_c);
        ffty_i2o(ffty_p);
        transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
            transp_c);

        solve_helmholtz_u4(
            &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
            nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
            ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
            nz, nghost_z,
            dx, dy, dz, nudt, 0.0, 0.0, bctype, bctype);

        transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
            transp_c);
        ffty_o2i(ffty_p);
        transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
            transp_c);
        fftx_c2d(fftx_p);
        transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
            transp_d);
        ghostz_pad_double(u1_3d, zwork_d, ghostz_p);
    }
    //Put bc back:
    apply_bc_u4(u1, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xu, yp, dz, bc_bot_case9, bc_top_case9, bctype, bctype);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = -3; iz < nz + 3; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                double err = u[ijk] - u1[ijk];
                if (fabs(err) > 1e-9) // Check forward-backward consistency
                                       // of Helmholtz operator/solver
                    printf("ix=%d iy=%d iz=%d u=%+e u1=%+e err=%+e\n",
                        ix, iy, iz, u[ijk], u1[ijk], err);
            }

    double *u_avg = (double *) malloc(sizeof(double) * n2);

    get_xy_avg(u1, u_avg, local_nx, local_ny, nz,
        nghost_x, nghost_y, nghost_z, nx, ny);
    if (nrank == 0)
        for (k = 0; k < n2; k++) printf("%+e\n", u_avg[k]);

    free(u_avg);

    free(xu);
    free(yp);
    free(dz);
    free(z);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d,   zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(ubc_3d, zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(u1_3d,  zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(h_3d,   zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(hbc_3d, zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);

    diffuseu_finalize4();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


void case10(int p_row, int p_col,
            int nx, int ny, int nz,
            int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    diffuseu_initialize4(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
                     - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***h_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *h = &(h_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    double *u_avg = (double *) malloc(sizeof(double) * n2);
    double *u_var = (double *) malloc(sizeof(double) * n2);
    check(u_avg != NULL);
    check(u_var != NULL);

    srand(nrank);

    for (i = 0; i < local_size; i++) {
        u[i] = 0.5 + (double) rand() / (double) RAND_MAX;
    }
    for (i = 0; i < local_size; i++) h[i] = FILL_VALUE;

    int step, it, nt = 5, istat = 5;
    double nu = 0.1, dt = 10.0;

    for (it = 0; it < nt; it++) {

        for (step = 0; step < 3; step++) { // Time marching of 4th-order u-diffusion

            double nuadt = nu * rk3_alp[step] * dt;
            double nubdt = nu * rk3_bet[step] * dt;

            ghostz_periodic(u, ghostz_p);

            helmholtz_operator_u4(u, h, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt);

            ghostz_truncate_double(h_3d, zwork_d, ghostz_p);
            transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
                transp_d);
            fftx_d2c(fftx_p);
            transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
                transp_c);
            ffty_i2o(ffty_p);
            transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                transp_c);

            solve_helmholtz_u4(
            &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt, 0.0, 0.0, DIRICHU, DIRICHU);

            transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
                transp_c);
            ffty_o2i(ffty_p);
            transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
                transp_c);
            fftx_c2d(fftx_p);
            transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
                transp_d);
            ghostz_pad_double(u_3d, zwork_d, ghostz_p);
        }

        double t = dt * (double) (it + 1);

        for (local_ix = 0; local_ix < local_nx; local_ix++)
            for (local_iy = 0; local_iy < local_ny; local_iy++)
                for (iz = 0; iz < nz; iz++) {
                    i = local_ix + nghost_x;
                    j = local_iy + nghost_y;
                    k = iz + nghost_z;
                    int ix = local_ix + decomp_d->zst[0] - 1;
                    int iy = local_iy + decomp_d->zst[1] - 1;
                    int ijk = (i * n1 + j) * n2 + k;
                    if (ix == 0 && iy == 0 && iz == nz / 2)
                        printf("%+e %+e %+e\n", t, z[k], u[ijk]);
                }

        if ((it + 1) % istat == 0) {
            get_xy_avg(u, u_avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            get_xy_corr(u, u, u_var,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            if (nrank == 0)
                for (k = 0; k < n2; k++)
                    printf("%+e %+e %+e %+e\n", t, z[k], u_avg[k], u_var[k]);
        }
    }

    free(dz);
    free(z);
    free(u_avg);
    free(u_var);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(h_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);

    diffuseu_finalize4();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


static double bc_bot_case11(double x, double y) { return x * y - 2.0; }
static double bc_top_case11(double x, double y) { return x + y + 1.0; }

void case11(int p_row, int p_col,
            int nx, int ny, int nz,
            int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    diffusew_initialize4(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *xp = (double *) malloc(sizeof(double) * n0);
    double *yp = (double *) malloc(sizeof(double) * n1);
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(xp != NULL);
    check(yp != NULL);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (i = 0; i < n0; i++) {
        local_ix = i - nghost_x;
        int ix = decomp_d->zst[0] - 1 + local_ix;
        ix = (ix + nx) % nx; // ix = 0, 1, ..., nx - 1
        xp[i] = ((double) (ix + 1) - 0.5) * dx;
    }

    for (j = 0; j < n1; j++) {
        local_iy = j - nghost_y;
        int iy = decomp_d->zst[1] - 1 + local_iy;
        iy = (iy + ny) % ny; // iy = 0, 1, ..., ny - 1
        yp[j] = ((double) (iy + 1) - 0.5) * dy;
    }

    for (k = 0; k < n2; k++) dz[k] = FILL_VALUE;
    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
                     - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = 1.0 / (double) nz;
    }
    //ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double   ***w_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double ***wbc_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double  ***w1_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double   ***h_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double ***hbc_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);

    double   *w =   &(w_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *wbc = &(wbc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double  *w1 =  &(w1_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double   *h =   &(h_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *hbc = &(hbc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    srand(nrank);

    for (i = 0; i < local_size; i++) w[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) w1[i] = FILL_VALUE;
    for (i = 0; i < local_size; i++) h[i] = FILL_VALUE;

    ghostz_periodic(w, ghostz_p);

    double bctype = DIRICHW;
    int test1d = 1;
    if (test1d == 1) {
        if (nrank == 0) printf("----Testing 1D helmholtz solver----\n");
    }
    else {
        if (nrank == 0) printf("----Testing 3D helmholtz solver----\n");
    }

    double nudt = 0.7;
    apply_bc_w4(w, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xp, yp, dz, bc_bot_case11, bc_top_case11, bctype, bctype);
    if (test1d == 1) {
        helmholtz1d_operator_w4(w, h, local_nx, local_ny, nz,
                                nghost_x, nghost_y, nghost_z, dz, nudt);
    }
    else {
        helmholtz_operator_w4(w, h, local_nx, local_ny, nz,
                              nghost_x, nghost_y, nghost_z, dx, dy, dz, nudt);
    }
    //Done setting h here.

    for (i = 0; i < local_size; i++) wbc[i] = 0.0;
    apply_bc_w4(wbc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xp, yp, dz, bc_bot_case11, bc_top_case11, bctype, bctype);
    if (test1d == 1) {
        helmholtz1d_operator_w4(wbc, hbc, local_nx, local_ny, nz,
                                nghost_x, nghost_y, nghost_z, dz, nudt);
    }
    else {
        helmholtz_operator_w4(wbc, hbc, local_nx, local_ny, nz,
                              nghost_x, nghost_y, nghost_z, dx, dy, dz, nudt);
    }
    for (i = 0; i < local_size; i++) h[i] -= hbc[i];

    if (test1d == 1) {
        solve_helmholtz1d_w4(w1, h, local_nx, local_ny, nz,
                             nghost_x, nghost_y, nghost_z, dz, nudt,
                             0.0, 0.0, bctype, bctype);
    }
    else {
        ghostz_truncate_double(h_3d, zwork_d, ghostz_p);
        transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
            transp_d);
        fftx_d2c(fftx_p);
        transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
            transp_c);
        ffty_i2o(ffty_p);
        transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
            transp_c);

        solve_helmholtz_w4(
            &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
            nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
            ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
            nz, nghost_z,
            dx, dy, dz, nudt, 0.0, 0.0, bctype, bctype);

        transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
            transp_c);
        ffty_o2i(ffty_p);
        transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
            transp_c);
        fftx_c2d(fftx_p);
        transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
            transp_d);
        ghostz_pad_double(w1_3d, zwork_d, ghostz_p);
    }
    //Put bc back:
    apply_bc_w4(w1, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xp, yp, dz, bc_bot_case11, bc_top_case11, bctype, bctype);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = -3; iz < nz + 2; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                int ijk = (i * n1 + j) * n2 + k;
                double err = w[ijk] - w1[ijk];
                if (fabs(err) > 1e-9) // Check forward-backward consistency
                                       // of Helmholtz operator/solver
                    printf("ix=%d iy=%d iz=%d w=%+e w1=%+e err=%+e\n",
                        ix, iy, iz, w[ijk], w1[ijk], err);
            }

    double *w_avg = (double *) malloc(sizeof(double) * n2);

    get_xy_avg(w1, w_avg, local_nx, local_ny, nz,
        nghost_x, nghost_y, nghost_z, nx, ny);
    if (nrank == 0)
        for (k = 0; k < n2; k++) printf("%+e\n", w_avg[k]);

    free(w_avg);

    free(xp);
    free(yp);
    free(dz);
    free(z);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(w_3d,   zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(wbc_3d, zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(w1_3d,  zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(h_3d,   zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(hbc_3d, zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);

    diffusew_finalize4();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


void case12(int p_row, int p_col,
            int nx, int ny, int nz,
            int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    diffusew_initialize4(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 0.5 * (cos(M_PI * (double) iz / (double) nz)
                     - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_w(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***w_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***h_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);

    double *w = &(w_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *h = &(h_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    double *w_avg = (double *) malloc(sizeof(double) * n2);
    double *w_var = (double *) malloc(sizeof(double) * n2);
    check(w_avg != NULL);
    check(w_var != NULL);

    srand(nrank);

    for (i = 0; i < local_size; i++) {
        w[i] = 0.5 + (double) rand() / (double) RAND_MAX;
    }
    for (i = 0; i < local_size; i++) h[i] = FILL_VALUE;

    int step, it, nt = 5, istat = 5;
    double nu = 0.1, dt = 10.0;

    for (it = 0; it < nt; it++) {

        for (step = 0; step < 3; step++) { // Time marching of 4th-order w-diffusion

            double nuadt = nu * rk3_alp[step] * dt;
            double nubdt = nu * rk3_bet[step] * dt;

            ghostz_periodic(w, ghostz_p);

            helmholtz_operator_w4(w, h, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt);

            ghostz_truncate_double(h_3d, zwork_d, ghostz_p);
            transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
                transp_d);
            fftx_d2c(fftx_p);
            transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
                transp_c);
            ffty_i2o(ffty_p);
            transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                transp_c);

            solve_helmholtz_w4(
            &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt, 0.0, 0.0, DIRICHW, DIRICHW);

            transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
                transp_c);
            ffty_o2i(ffty_p);
            transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
                transp_c);
            fftx_c2d(fftx_p);
            transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
                transp_d);
            ghostz_pad_double(w_3d, zwork_d, ghostz_p);
        }

        double t = dt * (double) (it + 1);

        for (local_ix = 0; local_ix < local_nx; local_ix++)
            for (local_iy = 0; local_iy < local_ny; local_iy++)
                for (iz = 0; iz < nz; iz++) {
                    i = local_ix + nghost_x;
                    j = local_iy + nghost_y;
                    k = iz + nghost_z;
                    int ix = local_ix + decomp_d->zst[0] - 1;
                    int iy = local_iy + decomp_d->zst[1] - 1;
                    int ijk = (i * n1 + j) * n2 + k;
                    if (ix == 0 && iy == 0 && iz == nz / 2)
                        printf("%+e %+e %+e\n", t, z[k], w[ijk]);
                }

        if ((it + 1) % istat == 0) {
            get_xy_avg(w, w_avg, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, nx, ny);
            get_xy_corr(w, w, w_var, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, nx, ny);
            if (nrank == 0)
                for (k = 0; k < n2; k++)
                    printf("%+e %+e %+e %+e\n", t, z[k], w_avg[k], w_var[k]);
        }
    }

    free(dz);
    free(z);
    free(w_avg);
    free(w_var);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(w_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(h_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);

    diffusew_finalize4();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


static double bc_u_bot_case13(double x, double y) { unused(x); unused(y); return 10.0; }
static double bc_u_top_case13(double x, double y) { unused(x); unused(y); return 0.0; }
static double bc_v_bot_case13(double x, double y) { unused(x); unused(y); return 10.0; }
static double bc_v_top_case13(double x, double y) { unused(x); unused(y); return 0.0; }
static double bc_w_bot_case13(double x, double y) { unused(x); unused(y); return 3.0; }
static double bc_w_top_case13(double x, double y) { unused(x); unused(y); return 3.0; }

void case13(int p_row, int p_col,
            int nx, int ny, int nz,
            int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    if (nrank == 0)
        printf("--------\nCheck 4th-order advective stress conservation\n--------\n");

    pressure_initialize4(nz, nghost_z);
    diffuseu_initialize4(nz, nghost_z);
    diffusew_initialize4(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *xp = (double *) malloc(sizeof(double) * n0);
    double *xu = (double *) malloc(sizeof(double) * n0);
    double *yp = (double *) malloc(sizeof(double) * n1);
    double *yv = (double *) malloc(sizeof(double) * n1);
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *zp = (double *) malloc(sizeof(double) * n2);
    check(xp != NULL);
    check(xu != NULL);
    check(yp != NULL);
    check(yv != NULL);
    check(dz != NULL);
    check(zp != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (i = 0; i < n0; i++) {
        local_ix = i - nghost_x;
        int ix = decomp_d->zst[0] - 1 + local_ix;
        ix = (ix + nx) % nx; // ix = 0, 1, ..., nx - 1
        xu[i] = (double) (ix + 1) * dx;
        xp[i] = ((double) (ix + 1) - 0.5) * dx;
    }

    for (j = 0; j < n1; j++) {
        local_iy = j - nghost_y;
        int iy = decomp_d->zst[1] - 1 + local_iy;
        iy = (iy + ny) % ny; // iy = 0, 1, ..., ny - 1
        yv[j] = (double) (iy + 1) * dy;
        yp[j] = ((double) (iy + 1) - 0.5) * dy;
    }

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 1.0 * 0.5 * (cos(M_PI * (double) iz / (double) nz)
                          - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, zp, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                       decomp_c->zst[1], decomp_c->zen[1],
                                       decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***v_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***w_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***d_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***p_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***muphx_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mvphx_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mwphx_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***muphy_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mvphy_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mwphy_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***muphz_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mvphz_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***mwphz_3d = d3tensor(zst_pad[0], zen_pad[0],
                                  zst_pad[1], zen_pad[1],
                                  zst_pad[2], zen_pad[2]);
    double ***fu_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***fv_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***fw_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *v = &(v_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *w = &(w_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *d = &(d_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *p = &(p_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *muphx = &(muphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mvphx = &(mvphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mwphx = &(mwphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *muphy = &(muphy_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mvphy = &(mvphy_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mwphy = &(mwphy_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *muphz = &(muphz_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mvphz = &(mvphz_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mwphz = &(mwphz_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fu = &(fu_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fv = &(fv_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fw = &(fw_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    srand(nrank);
 
    for (i = 0; i < local_size; i++) u[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) v[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) w[i] = (double) rand() / (double) RAND_MAX;

    ghostz_periodic(u, ghostz_p);
    apply_bc_u4(u, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xu, yp, dz, bc_u_bot_case13, bc_u_top_case13, DIRICHU, DIRICHU);
    ghostz_periodic(v, ghostz_p);
    apply_bc_u4(v, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xp, yv, dz, bc_v_bot_case13, bc_v_top_case13, DIRICHU, DIRICHU);
    ghostz_periodic(w, ghostz_p);
    apply_bc_w4(w, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                xp, yp, dz, bc_w_bot_case13, bc_w_top_case13, DIRICHW, DIRICHW);
    divergence4(u, v, w, d, local_nx, local_ny, nz,
                nghost_x, nghost_y, nghost_z, dx, dy, dz);
    ghostz_truncate_double(d_3d, zwork_d, ghostz_p);
    transpose_z_to_x(
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
        transp_d);
    fftx_d2c(fftx_p);
    transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_i2o(ffty_p);
    transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        transp_c);

    pressure_poisson4(
        &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
        ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
        nz, nghost_z,
        dx, dy, dz, 1.0);

    transpose_z_to_y(
               &(zwork_c[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_o2i(ffty_p);
    transpose_y_to_x(
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        transp_c);
    fftx_c2d(fftx_p);
    transpose_x_to_z(
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
               &(zwork_d[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        transp_d);
    ghostz_pad_double(p_3d, zwork_d, ghostz_p);
    ghostz_periodic(p, ghostz_p);
    project4(u, v, w, p, local_nx, local_ny, nz,
             nghost_x, nghost_y, nghost_z, dx, dy, dz, 1.0);
    ghostz_periodic(u, ghostz_p);
    ghostz_periodic(v, ghostz_p);
    ghostz_periodic(w, ghostz_p);

    massfluxx4(u, v, w, muphx, mvphx, mwphx,
               local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);
    advectu4(u, muphx, mvphx, mwphx, fu,
             local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);

    massfluxy4(u, v, w, muphy, mvphy, mwphy,
               local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);
    advectv4(v, muphy, mvphy, mwphy, fv,
             local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);

    massfluxz4(u, v, w, muphz, mvphz, mwphz,
               local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);
    advectw4(w, muphz, mvphz, mwphz, fw,
             local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, dx, dy, dz);

    double ksumu = 0.0;
    double ksumv = 0.0;
    double ksumw = 0.0;
    double msumu = 0.0;
    double msumv = 0.0;
    double msumw = 0.0;

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ijk = (i * n1 + j) * n2 + k;
                ksumu +=  u[ijk] * dx * dy * dz[k] * fu[ijk];
                ksumv +=  v[ijk] * dx * dy * dz[k] * fv[ijk];
                msumu +=           dx * dy * dz[k] * fu[ijk];
                msumv +=           dx * dy * dz[k] * fv[ijk];
            }

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz - 1; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ijk = (i * n1 + j) * n2 + k;
                ksumw +=  w[ijk] * dx * dy * 0.5 * (dz[k] + dz[k + 1]) * fw[ijk];
                msumw +=           dx * dy * 0.5 * (dz[k] + dz[k + 1]) * fw[ijk];
            }

    MPI_Allreduce(MPI_IN_PLACE, &ksumu, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ksumv, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ksumw, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &msumu, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &msumv, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &msumw, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (nrank == 0) {
        // Can check -(78) Sanderse (2014) J. Comput. Phys. 257:1472-1505.
        // -(76) cannot be checked because the boundary tendencies are never computed.
        printf("ksumu = %+e, ksumv = %+e, ksumw = %+e\n", ksumu, ksumv, ksumw);
        printf("msumu = %+e, msumv = %+e, msumw = %+e\n", msumu, msumv, msumw);
    }

    free(xp);
    free(xu);
    free(yp);
    free(yv);
    free(dz);
    free(zp);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c, decomp_c->zst[0], decomp_c->zen[0],
                           decomp_c->zst[1], decomp_c->zen[1],
                           decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(v_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(w_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(d_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(p_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(muphx_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mvphx_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mwphx_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(muphy_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mvphy_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mwphy_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(muphz_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mvphz_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(mwphz_3d, zst_pad[0], zen_pad[0],
                            zst_pad[1], zen_pad[1],
                            zst_pad[2], zen_pad[2]);
    free_d3tensor(fu_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(fv_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(fw_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);

    pressure_finalize4();
    diffuseu_finalize4();
    diffusew_finalize4();

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}


void case14(int p_row, int p_col,
            int nx, int ny, int nz,
            int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = 1.0 / (double) nx;
    double dy = 1.0 / (double) ny;
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *z = (double *) malloc(sizeof(double) * n2);
    check(dz != NULL);
    check(z != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (iz = 0; iz < nz; iz++) {
        int k = iz + nghost_z;
        dz[k] = 1.0 / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, z, nz, nghost_z);

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***u1_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *u1 = &(u1_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    for (i = 0; i < local_size; i++) u[i] = u1[i] = 0.0;

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                double x = (double) ix * dx;
                double y = (double) iy * dy;
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = x * y * z[k];
            }

    if (nrank == 0) printf("Writing temporary file\n");
    ghostz_truncate_double(u_3d, zwork_d, ghostz_p);
    io_write(zwork_d, "tmp.dat", nx, ny, nz,
        nghost_z, decomp_d->zst, decomp_d->zsz);

    for (i = decomp_d->zst[0]; i <= decomp_d->zen[0]; i++)
        for (j = decomp_d->zst[1]; j <= decomp_d->zen[1]; j++)
            for (k = decomp_d->zst[2]; k <= decomp_d->zen[2]; k++) {
                zwork_d[i][j][k] = 0.0;
            }

    if (nrank == 0) printf("Reading temporary file\n");
    io_read(zwork_d, "tmp.dat", nx, ny, nz,
        nghost_z, decomp_d->zst, decomp_d->zsz);
    ghostz_pad_double(u1_3d, zwork_d, ghostz_p);

    if (nrank == 0 && 0 != remove("tmp.dat")) {
        fprintf(stderr, "Cannot delete temporary file\n");
    }

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                //int ix = local_ix + decomp_d->zst[0] - 1;
                //int iy = local_iy + decomp_d->zst[1] - 1;
                //double x = (double) ix * dx;
                //double y = (double) iy * dy;
                int ijk = (i * n1 + j) * n2 + k;
                double err = u[ijk] - u1[ijk];
                if (fabs(err) > DBL_EPSILON)
                    printf("%+e %+e %+e\n", err, u[ijk], u1[ijk]);
            }

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(u1_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);

    destroy_ghostz_plan(ghostz_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
}


void case15(int p_row, int p_col,
            int nx, int ny, int nz,
            int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    if (nrank == 0)
        printf("----\nCheck consistency of get_xy_moment()\n----\n");

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int local_size = n0 * n1 * n2;

    double ***zwork_d = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                 decomp_d->zst[1], decomp_d->zen[1],
                                 decomp_d->zst[2], decomp_d->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    double *corr = (double *) malloc(sizeof(double) * n2);
    double *mom1 = (double *) malloc(sizeof(double) * n2);
    double *mom2 = (double *) malloc(sizeof(double) * n2);
    double *mom3 = (double *) malloc(sizeof(double) * n2);
    double *mom4 = (double *) malloc(sizeof(double) * n2);

    check(corr != NULL);
    check(mom1 != NULL);
    check(mom2 != NULL);
    check(mom3 != NULL);
    check(mom4 != NULL);

    // Fill u[] with data from file or random numbers.

    //io_read(zwork_d, "outputdir/u_it10000.dat", nx, ny, nz,
    //    nghost_z, decomp_d->zst, decomp_d->zsz);
    //ghostz_pad_double(u_3d, zwork_d, ghostz_p);

    srand(nrank);
    int i;
    for (i = 0; i < local_size; i++) u[i] = (double) rand() / (double) RAND_MAX;

    ghostz_periodic(u, ghostz_p);

    get_xy_corr(u, u, corr,
        local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
    get_xy_moment(u, mom1, 1.0,
        local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
    get_xy_moment(u, mom2, 2.0,
        local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
    get_xy_moment(u, mom3, 3.0,
        local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
    get_xy_moment(u, mom4, 4.0,
        local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);

    int k;
    for (k = 0; k < n2; k++) {
        check(fabs(mom1[k]) < 1e-13);
        check(fabs(mom2[k] - corr[k]) < 1e-13);
    }

    //io_write_ascii(mom1, n2, "outputdir/u_mom1_it10000.dat");
    //io_write_ascii(mom2, n2, "outputdir/u_mom2_it10000.dat");
    //io_write_ascii(mom3, n2, "outputdir/u_mom3_it10000.dat");
    //io_write_ascii(mom4, n2, "outputdir/u_mom4_it10000.dat");

    free(corr);
    free(mom1);
    free(mom2);
    free(mom3);
    free(mom4);

    free_d3tensor(zwork_d, decomp_d->zst[0], decomp_d->zen[0],
                           decomp_d->zst[1], decomp_d->zen[1],
                           decomp_d->zst[2], decomp_d->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);

    destroy_ghostz_plan(ghostz_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
}


void case16(int p_row, int p_col,
            int nx, int ny, int nz,
            int nghost_x, int nghost_y, int nghost_z)
{
    int nrank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    check(nproc == p_row * p_col);

    int dims[2];
    dims[0] = p_row;
    dims[1] = p_col;
    int periodic[2];
    periodic[0] = 1;
    periodic[1] = 1;
    int coord[2];

    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, nrank, 2, coord);

    MPI_Datatype mpi_fftw_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_fftw_complex);
    MPI_Type_commit(&mpi_fftw_complex);

    decomp_plan *decomp_d = create_decomp_plan(dims, coord,
        nx, ny, nz + 2 * nghost_z);
    transpose_plan *transp_d = create_transpose_plan(dims,
        &comm_cart, MPI_DOUBLE,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_d->yst, decomp_d->yen, decomp_d->ysz,
        decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        decomp_d->x1st, decomp_d->x1en, decomp_d->x1dist,
        decomp_d->y1st, decomp_d->y1en, decomp_d->y1dist,
        decomp_d->y2st, decomp_d->y2en, decomp_d->y2dist,
        decomp_d->z2st, decomp_d->z2en, decomp_d->z2dist);

    decomp_plan *decomp_c = create_decomp_plan(dims, coord,
        nx / 2 + 1, ny, nz + 2 * nghost_z);
    transpose_plan *transp_c = create_transpose_plan(dims,
        &comm_cart, mpi_fftw_complex,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz,
        decomp_c->zst, decomp_c->zen, decomp_c->zsz,
        decomp_c->x1st, decomp_c->x1en, decomp_c->x1dist,
        decomp_c->y1st, decomp_c->y1en, decomp_c->y1dist,
        decomp_c->y2st, decomp_c->y2en, decomp_c->y2dist,
        decomp_c->z2st, decomp_c->z2en, decomp_c->z2dist);

    fftx_plan *fftx_p = create_fftx_plan(nx,
        decomp_d->xst, decomp_d->xen, decomp_d->xsz,
        decomp_c->xst, decomp_c->xen, decomp_c->xsz);
    ffty_plan *ffty_p = create_ffty_plan(ny,
        decomp_c->yst, decomp_c->yen, decomp_c->ysz);

    ghostz_plan *ghostz_p = create_ghostz_plan(&comm_cart,
        MPI_DOUBLE, decomp_d->zst, decomp_d->zen, decomp_d->zsz,
        nghost_x, nghost_y);

    if (nrank == 0)
        printf("----\nTesting get_xy_spec()\n----\n");

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int local_size = n0 * n1 * n2;

    double ***zwork_d1 = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                  decomp_d->zst[1], decomp_d->zen[1],
                                  decomp_d->zst[2], decomp_d->zen[2]);
    double ***zwork_d2 = d3tensor(decomp_d->zst[0], decomp_d->zen[0],
                                  decomp_d->zst[1], decomp_d->zen[1],
                                  decomp_d->zst[2], decomp_d->zen[2]);
    fftw_complex ***zwork_c1 = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                        decomp_c->zst[1], decomp_c->zen[1],
                                        decomp_c->zst[2], decomp_c->zen[2]);
    fftw_complex ***zwork_c2 = c3tensor(decomp_c->zst[0], decomp_c->zen[0],
                                        decomp_c->zst[1], decomp_c->zen[1],
                                        decomp_c->zst[2], decomp_c->zen[2]);

    int zst_pad[3] = {ghostz_p->zst_pad[0],
                      ghostz_p->zst_pad[1],
                      ghostz_p->zst_pad[2]};
    int zen_pad[3] = {ghostz_p->zen_pad[0],
                      ghostz_p->zen_pad[1],
                      ghostz_p->zen_pad[2]};

    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***v_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);

    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *v = &(v_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    int specx_size = (nx / 2 + 1) * n2 * 2;
    int specy_size = (ny / 2 + 1) * n2 * 2;

    double *specx = (double *) malloc(sizeof(double) * specx_size);
    double *specy = (double *) malloc(sizeof(double) * specy_size);
    double *corr = (double *) malloc(sizeof(double) * n2);
    double *sumcospecx = (double *) malloc(sizeof(double) * n2);
    double *sumqdspecx = (double *) malloc(sizeof(double) * n2);
    double *sumcospecy = (double *) malloc(sizeof(double) * n2);
    double *sumqdspecy = (double *) malloc(sizeof(double) * n2);

    check(specx != NULL);
    check(specy != NULL);
    check(corr != NULL);
    check(sumcospecx != NULL);
    check(sumqdspecx != NULL);
    check(sumcospecy != NULL);
    check(sumqdspecy != NULL);

    int local_ix, local_iy, iz, i, j, k;

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ix = local_ix + decomp_d->zst[0] - 1;
                int iy = local_iy + decomp_d->zst[1] - 1;
                double x = (double) ix / (double) nx;
                double y = (double) iy / (double) ny;
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = sin(2.0 * M_PI * x);
                v[ijk] = cos(2.0 * M_PI * y);
            }

    srand(nrank);

    for (i = 0; i < local_size; i++) u[i] = (double) rand() / (double) RAND_MAX;
    for (i = 0; i < local_size; i++) v[i] = (double) rand() / (double) RAND_MAX;

    ghostz_periodic(u, ghostz_p);
    ghostz_periodic(v, ghostz_p);

    ghostz_truncate_double(u_3d, zwork_d1, ghostz_p);
    transpose_z_to_x(
              &(zwork_d1[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
        transp_d);
    fftx_d2c(fftx_p);
    transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_i2o(ffty_p);
    transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
              &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        transp_c);

    ghostz_truncate_double(v_3d, zwork_d2, ghostz_p);
    transpose_z_to_x(
              &(zwork_d2[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]),
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]),
        transp_d);
    fftx_d2c(fftx_p);
    transpose_x_to_y(
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]),
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
        transp_c);
    ffty_i2o(ffty_p);
    transpose_y_to_z(
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]),
              &(zwork_c2[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        transp_c);

    get_xy_spec(
        &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        &(zwork_c2[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
        specx, specy,
        nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
        ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
        nz, nghost_z);
    get_xy_corr(u, v, corr, local_nx, local_ny, nz,
        nghost_x, nghost_y, nghost_z, nx, ny);

    for (k = 0; k < n2; k++) {
        sumcospecx[k] = sumqdspecx[k] = sumcospecy[k] = sumqdspecy[k] = 0.0;
    }

    if (nx % 2 == 0) // nx is even
        for (i = 0; i < nx / 2 + 1; i++)
            for (k = 0; k < n2; k++) {
                int ik = (i * n2 + k) * 2;
                if (i == 0 || i == nx / 2) {
                    sumcospecx[k] += specx[ik];
                    sumqdspecx[k] += specx[ik + 1];
                }
                else {
                    // One-dimensional cospectrum is even:
                    sumcospecx[k] += 2.0 * specx[ik];
                    // One-dimensional quadrature spectrum is odd:
                    sumqdspecx[k] += specx[ik + 1] - specx[ik + 1];
                }
            }
    else // nx is odd
        for (i = 0; i < nx / 2 + 1; i++)
            for (k = 0; k < n2; k++) {
                int ik = (i * n2 + k) * 2;
                if (i == 0) {
                    sumcospecx[k] += specx[ik];
                    sumqdspecx[k] += specx[ik + 1];
                }
                else {
                    // One-dimensional cospectrum is even:
                    sumcospecx[k] += 2.0 * specx[ik];
                    // One-dimensional quadrature spectrum is odd:
                    sumqdspecx[k] += specx[ik + 1] - specx[ik + 1];
                }
            }

    if (ny % 2 == 0) // ny is even
        for (i = 0; i < ny / 2 + 1; i++)
            for (k = 0; k < n2; k++) {
                int ik = (i * n2 + k) * 2;
                if (i == 0 || i == ny / 2) {
                    sumcospecy[k] += specy[ik];
                    sumqdspecy[k] += specy[ik + 1];
                }
                else {
                    // One-dimensional cospectrum is even:
                    sumcospecy[k] += 2.0 * specy[ik];
                    // One-dimensional quadrature spectrum is odd:
                    sumqdspecy[k] += specy[ik + 1] - specy[ik + 1];
                }
            }
    else // ny is odd
        for (i = 0; i < ny / 2 + 1; i++)
            for (k = 0; k < n2; k++) {
                int ik = (i * n2 + k) * 2;
                if (i == 0) {
                    sumcospecy[k] += specy[ik];
                    sumqdspecy[k] += specy[ik + 1];
                }
                else {
                    // One-dimensional cospectrum is even:
                    sumcospecy[k] += 2.0 * specy[ik];
                    // One-dimensional quadrature spectrum is odd:
                    sumqdspecy[k] += specy[ik + 1] - specy[ik + 1];
                }
            }

    if (nrank == 0) {
        for (k = 0; k < n2; k++) {
            if (fabs(corr[k] - sumcospecx[k]) > 1e-15 ||
                fabs(          sumqdspecx[k]) > 1e-15 ||
                fabs(corr[k] - sumcospecy[k]) > 1e-15 ||
                fabs(          sumqdspecy[k]) > 1e-15)
                printf("%d %+e %+e %+e %+e\n", k,
                    corr[k] - sumcospecx[k], sumqdspecx[k],
                    corr[k] - sumcospecy[k], sumqdspecy[k]);
        }
    }

    free(specx);
    free(specy);
    free(corr);
    free(sumcospecx);
    free(sumqdspecx);
    free(sumcospecy);
    free(sumqdspecy);

    free_d3tensor(zwork_d1, decomp_d->zst[0], decomp_d->zen[0],
                            decomp_d->zst[1], decomp_d->zen[1],
                            decomp_d->zst[2], decomp_d->zen[2]);
    free_d3tensor(zwork_d2, decomp_d->zst[0], decomp_d->zen[0],
                            decomp_d->zst[1], decomp_d->zen[1],
                            decomp_d->zst[2], decomp_d->zen[2]);
    free_c3tensor(zwork_c1, decomp_c->zst[0], decomp_c->zen[0],
                            decomp_c->zst[1], decomp_c->zen[1],
                            decomp_c->zst[2], decomp_c->zen[2]);
    free_c3tensor(zwork_c2, decomp_c->zst[0], decomp_c->zen[0],
                            decomp_c->zst[1], decomp_c->zen[1],
                            decomp_c->zst[2], decomp_c->zen[2]);

    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(v_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);

    destroy_ghostz_plan(ghostz_p);

    destroy_fftx_plan(fftx_p);
    destroy_ffty_plan(ffty_p);

    destroy_transpose_plan(transp_d);
    destroy_decomp_plan(decomp_d);
    destroy_transpose_plan(transp_c);
    destroy_decomp_plan(decomp_c);

    MPI_Type_free(&mpi_fftw_complex);
}
