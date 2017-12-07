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
#include "fourth.h"
#include "pressure.h"
#include "diffuseu.h"
#include "diffusew.h"
#include "advect.h"
#include "scalar.h"
#include "channel.h"


#define FFT3D_FWD(A, WORKD, WORKC) { \
    ghostz_truncate_double(A, WORKD, ghostz_p); \
    transpose_z_to_x( \
                 &(WORKD[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]), \
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]), \
        transp_d); \
    fftx_d2c(fftx_p); \
    transpose_x_to_y( \
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]), \
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]), \
        transp_c); \
    ffty_i2o(ffty_p); \
    transpose_y_to_z( \
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]), \
                 &(WORKC[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]), \
        transp_c); \
}

#define FFT3D_BWD(A, WORKD, WORKC) { \
    transpose_z_to_y( \
                 &(WORKC[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]), \
        &(ffty_p->work_o[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]), \
        transp_c); \
    ffty_o2i(ffty_p); \
    transpose_y_to_x( \
        &(ffty_p->work_i[decomp_c->yst[0]][decomp_c->yst[1]][decomp_c->yst[2]]), \
        &(fftx_p->work_c[decomp_c->xst[0]][decomp_c->xst[1]][decomp_c->xst[2]]), \
        transp_c); \
    fftx_c2d(fftx_p); \
    transpose_x_to_z( \
        &(fftx_p->work_d[decomp_d->xst[0]][decomp_d->xst[1]][decomp_d->xst[2]]), \
                 &(WORKD[decomp_d->zst[0]][decomp_d->zst[1]][decomp_d->zst[2]]), \
        transp_d); \
    ghostz_pad_double(A, WORKD, ghostz_p); \
}


static double uframe;
static double vframe;

static double bc_u_bot(double x, double y) { unused(x); unused(y); return 0.0 - uframe; }
static double bc_u_top(double x, double y) { unused(x); unused(y); return 0.0 - uframe; }
static double bc_v_bot(double x, double y) { unused(x); unused(y); return 0.0 - vframe; }
static double bc_v_top(double x, double y) { unused(x); unused(y); return 0.0 - vframe; }
static double bc_w_bot(double x, double y) { unused(x); unused(y); return 0.0; }
static double bc_w_top(double x, double y) { unused(x); unused(y); return 0.0; }
static double bc_c_bot(double x, double y) { unused(x); unused(y); return -0.5; }
static double bc_c_top(double x, double y) { unused(x); unused(y); return +0.5; }


void channel2(int p_row, int p_col,
              int nx, int ny, int nz, int it, int nt, int istat,
              int nghost_x, int nghost_y, int nghost_z,
              double lx, double ly, double lz,
              double dt, double want_cfl, int cflmode,
              double nu, double prandtl,
              double dpdx, double want_uvolavg, int uvolavgmode,
              double want_cvolavg, int cvolavgmode,
              double setuframe, double setvframe,
              double betg_x, double betg_y, double betg_z)
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
    diffusew_initialize2(nz, nghost_z);
    pressure_initialize2(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = lx / (double) nx;
    double dy = ly / (double) ny;
    double *xp = (double *) malloc(sizeof(double) * n0);
    double *xu = (double *) malloc(sizeof(double) * n0);
    double *yp = (double *) malloc(sizeof(double) * n1);
    double *yv = (double *) malloc(sizeof(double) * n1);
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *zp = (double *) malloc(sizeof(double) * n2);
    double *zw = (double *) malloc(sizeof(double) * n2);
    check(xp != NULL);
    check(xu != NULL);
    check(yp != NULL);
    check(yv != NULL);
    check(dz != NULL);
    check(zp != NULL);
    check(zw != NULL);

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
        dz[k] = lz * 0.5 * (cos(M_PI * (double) iz / (double) nz)
                          - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = lz / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, zp, nz, nghost_z);
    get_z_at_w(dz, zw, nz, nghost_z);

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

    double ***bc_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***hbc_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***v_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***w_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***c_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***p_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***h_3d = d3tensor(zst_pad[0], zen_pad[0],
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
    double ***fu_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***fv_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***fw_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***fc_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***gu_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***gv_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***gw_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***gc_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);

    double *bc = &(bc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *hbc = &(hbc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *v = &(v_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *w = &(w_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *c = &(c_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *p = &(p_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *h = &(h_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *muphx = &(muphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mvphx = &(mvphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mwphx = &(mwphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *muphy = muphx;
    double *mvphy = mvphx;
    double *mwphy = mwphx;
    double *muphz = muphx;
    double *mvphz = mvphx;
    double *mwphz = mwphx;
    double *mucphx = muphx;
    double *mvcphy = mvphx;
    double *mwcphz = mwphx;
    double *fu = &(fu_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fv = &(fv_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fw = &(fw_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fc = &(fc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *gu = &(gu_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *gv = &(gv_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *gw = &(gw_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *gc = &(gc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    int specx_size = (nx / 2 + 1) * n2 * 2;
    int specy_size = (ny / 2 + 1) * n2 * 2;
    int xavg_size = ny * n2;
    int xcorr_size = ny * n2;
    //int yavg_size = n0 * n2;
    //int ycorr_size = n0 * n2;

    double *avg = (double *) malloc(sizeof(double) * n2);
    double *corr = (double *) malloc(sizeof(double) * n2);
    double *mom3 = (double *) malloc(sizeof(double) * n2);
    double *mom4 = (double *) malloc(sizeof(double) * n2);
    double *specx = (double *) malloc(sizeof(double) * specx_size);
    double *specy = (double *) malloc(sizeof(double) * specy_size);
    double *xavg = (double *) malloc(sizeof(double) * xavg_size);
    double *xcorr = (double *) malloc(sizeof(double) * xcorr_size);
    //double *yavg = (double *) malloc(sizeof(double) * yavg_size);
    //double *ycorr = (double *) malloc(sizeof(double) * ycorr_size);

    check(avg != NULL);
    check(corr != NULL);
    check(mom3 != NULL);
    check(mom4 != NULL);
    check(specx != NULL);
    check(specy != NULL);
    check(xavg != NULL);
    check(xcorr != NULL);
    //check(yavg != NULL);
    //check(ycorr != NULL);

    srand(nrank);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = 0.0;
                v[ijk] = 0.0;
                w[ijk] = 0.0;
                c[ijk] = 0.0;
            }

    for (i = 0; i < local_size; i++) hbc[i] = p[i] = h[i]
        = fu[i] = fv[i] = fw[i] = fc[i]
        = gu[i] = gv[i] = gw[i] = gc[i] = FILL_VALUE;

    char fname[100];

    sprintf(fname, "outputdir/u_it%d.dat", it);
    io_read(zwork_d1, fname, nx, ny, nz,
        nghost_z, decomp_d->zst, decomp_d->zsz);
    ghostz_pad_double(u_3d, zwork_d1, ghostz_p);
    sprintf(fname, "outputdir/v_it%d.dat", it);
    io_read(zwork_d1, fname, nx, ny, nz,
        nghost_z, decomp_d->zst, decomp_d->zsz);
    ghostz_pad_double(v_3d, zwork_d1, ghostz_p);
    sprintf(fname, "outputdir/w_it%d.dat", it);
    io_read(zwork_d1, fname, nx, ny, nz,
        nghost_z, decomp_d->zst, decomp_d->zsz);
    ghostz_pad_double(w_3d, zwork_d1, ghostz_p);
    sprintf(fname, "outputdir/c_it%d.dat", it);
    io_read(zwork_d1, fname, nx, ny, nz,
        nghost_z, decomp_d->zst, decomp_d->zsz);
    ghostz_pad_double(c_3d, zwork_d1, ghostz_p);

    uframe = setuframe;
    vframe = setvframe;
    double bctype_u_bot = DIRICHU;
    double bctype_u_top = DIRICHU;
    double bctype_v_bot = DIRICHU;
    double bctype_v_top = DIRICHU;
    double bctype_c_bot = DIRICHU;
    double bctype_c_top = DIRICHU;
    double bctype_w_bot = DIRICHW;
    double bctype_w_top = DIRICHW;

    int step;

    for ( ; it < nt; it++) {

        double cfl = get_cfl(u, v, w,
                             local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                             dx, dy, dz, dt);
        if (cflmode == 1) { // Make CFL = want_cfl
            dt *= cfl > 0.0 ? want_cfl / cfl : 1.0;
            cfl = cfl > 0.0 ? want_cfl       : cfl;
        }
        io_append_ascii((double []) { (double) it,
            dt, cfl }, 3, "outputdir/cfl.dat");

        for (step = 0; step < 3; step++) { // RK3 stepper

            {
                double *tu = fu; fu = gu; gu = tu; // Swap
                double *tv = fv; fv = gv; gv = tv;
                double *tw = fw; fw = gw; gw = tw;
                double *tc = fc; fc = gc; gc = tc;
            }

            ghostz_periodic(u, ghostz_p);
            apply_bc_u2(u, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xu, yp, dz, bc_u_bot, bc_u_top, bctype_u_bot, bctype_u_top);

            ghostz_periodic(v, ghostz_p);
            apply_bc_u2(v, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yv, dz, bc_v_bot, bc_v_top, bctype_v_bot, bctype_v_top);

            ghostz_periodic(c, ghostz_p);
            apply_bc_u2(c, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_c_bot, bc_c_top, bctype_c_bot, bctype_c_top);

            ghostz_periodic(w, ghostz_p);
            apply_bc_w2(w, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_w_bot, bc_w_top, bctype_w_bot, bctype_w_top);

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

            flux_quick(u, v, w, c, mucphx, mvcphy, mwcphz, local_nx, local_ny, nz,
                       nghost_x, nghost_y, nghost_z, dx, dy, dz);

            advect_scalar(mucphx, mvcphy, mwcphz, fc, local_nx, local_ny, nz,
                          nghost_x, nghost_y, nghost_z, dx, dy, dz);

            add_buoyancy2(c, fu, fv, fw, betg_x, betg_y, betg_z,
                          local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z);

            ghostz_periodic(fu, ghostz_p);
            ghostz_periodic(fv, ghostz_p);
            ghostz_periodic(fw, ghostz_p);
            ghostz_periodic(fc, ghostz_p);

            double nuadt = nu * rk3_alp[step] * dt;
            double nubdt = nu * rk3_bet[step] * dt;
            double abdt = (rk3_alp[step] + rk3_bet[step]) * dt;
            double gamdt = rk3_gam[step] * dt;
            double zetdt = rk3_zet[step] * dt;

            apply_bc_u2(u, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xu, yp, dz, bc_u_bot, bc_u_top, bctype_u_bot, bctype_u_top);
            helmholtz_operator_u2(u, h, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt);
            for (i = 0; i < local_size; i++) {
                h[i] += gamdt * fu[i] + zetdt * gu[i];
            }
            for (i = 0; i < local_size; i++) bc[i] = 0.0;
            apply_bc_u2(bc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xu, yp, dz, bc_u_bot, bc_u_top, bctype_u_bot, bctype_u_top);
            helmholtz_operator_u2(bc, hbc, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, -nubdt);
            for (i = 0; i < local_size; i++) h[i] -= hbc[i];

            FFT3D_FWD(h_3d, zwork_d1, zwork_c1);

            solve_helmholtz_u2(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt, 0.0, 0.0, bctype_u_bot, bctype_u_top);

            FFT3D_BWD(u_3d, zwork_d1, zwork_c1);

            ghostz_periodic(u, ghostz_p);
            apply_bc_u2(u, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xu, yp, dz, bc_u_bot, bc_u_top, bctype_u_bot, bctype_u_top);

            // Begin u statistics
            get_xy_avg(u, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            double uvolavg0 = get_z_avg(avg, nz, nghost_z, dz);
            double nududzbot0 = nu *
                (avg[nghost_z] - avg[nghost_z - 1]) /
                (0.5 * (dz[nghost_z] + dz[nghost_z - 1]));
            double nududztop0 = nu *
                (avg[n2 - nghost_z] - avg[n2 - nghost_z - 1]) /
                (0.5 * (dz[n2 - nghost_z] + dz[n2 - nghost_z - 1]));
            double ucenterline0 = nz % 2 == 0 ? 0.5 * (avg[nghost_z + nz / 2 - 1] + avg[nghost_z + nz / 2])
                                                     : avg[nghost_z + nz / 2];

            for (i = 0; i < local_size; i++) h[i] = gamdt;
            solve_helmholtz1d_u2(h, h, local_nx, local_ny, nz,
                                 nghost_x, nghost_y, nghost_z, dz, -nubdt,
                                 0.0, 0.0, bctype_u_bot, bctype_u_top);

            get_xy_avg(h, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            double uvolavg1 = get_z_avg(avg, nz, nghost_z, dz);
            double nududzbot1 = nu *
                (avg[nghost_z] - avg[nghost_z - 1]) /
                (0.5 * (dz[nghost_z] + dz[nghost_z - 1]));
            double nududztop1 = nu *
                (avg[n2 - nghost_z] - avg[n2 - nghost_z - 1]) /
                (0.5 * (dz[n2 - nghost_z] + dz[n2 - nghost_z - 1]));
            double ucenterline1 = nz % 2 == 0 ? 0.5 * (avg[nghost_z + nz / 2 - 1] + avg[nghost_z + nz / 2])
                                                     : avg[nghost_z + nz / 2];

            double mdpdx = uvolavgmode == 1 ? (want_uvolavg - uframe - uvolavg0) / uvolavg1 : -dpdx;

            uvolavg0 += uvolavg1 * mdpdx;
            nududzbot0 += nududzbot1 * mdpdx;
            nududztop0 += nududztop1 * mdpdx;
            ucenterline0 += ucenterline1 * mdpdx;
            for (i = 0; i < local_size; i++) {
                u[i] += h[i] * mdpdx;
                fu[i] += mdpdx;
            }

            if (step == 2)
                io_append_ascii( (double []) { (double) (it + 1),
                    mdpdx, nududzbot0, nududztop0, uvolavg0, ucenterline0 },
                    6, "outputdir/pstat.dat");
            // End u statistics

            apply_bc_u2(v, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yv, dz, bc_v_bot, bc_v_top, bctype_v_bot, bctype_v_top);
            helmholtz_operator_u2(v, h, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt);
            for (i = 0; i < local_size; i++) {
                h[i] += gamdt * fv[i] + zetdt * gv[i];
            }
            for (i = 0; i < local_size; i++) bc[i] = 0.0;
            apply_bc_u2(bc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yv, dz, bc_v_bot, bc_v_top, bctype_v_bot, bctype_v_top);
            helmholtz_operator_u2(bc, hbc, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, -nubdt);
            for (i = 0; i < local_size; i++) h[i] -= hbc[i];

            FFT3D_FWD(h_3d, zwork_d1, zwork_c1);

            solve_helmholtz_u2(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt, 0.0, 0.0, bctype_v_bot, bctype_v_top);

            FFT3D_BWD(v_3d, zwork_d1, zwork_c1);

            ghostz_periodic(v, ghostz_p);
            apply_bc_u2(v, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yv, dz, bc_v_bot, bc_v_top, bctype_v_bot, bctype_v_top);

            apply_bc_u2(c, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_c_bot, bc_c_top, bctype_c_bot, bctype_c_top);
            helmholtz_operator_u2(c, h, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt / prandtl);
            for (i = 0; i < local_size; i++) {
                h[i] += gamdt * fc[i] + zetdt * gc[i];
            }
            for (i = 0; i < local_size; i++) bc[i] = 0.0;
            apply_bc_u2(bc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_c_bot, bc_c_top, bctype_c_bot, bctype_c_top);
            helmholtz_operator_u2(bc, hbc, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, -nubdt / prandtl);
            for (i = 0; i < local_size; i++) h[i] -= hbc[i];

            FFT3D_FWD(h_3d, zwork_d1, zwork_c1);

            solve_helmholtz_u2(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt / prandtl, 0.0, 0.0, bctype_c_bot, bctype_c_top);

            FFT3D_BWD(c_3d, zwork_d1, zwork_c1);

            ghostz_periodic(c, ghostz_p);
            apply_bc_u2(c, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_c_bot, bc_c_top, bctype_c_bot, bctype_c_top);

            // Begin c statistics
            get_xy_avg(c, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            double cvolavg0 = get_z_avg(avg, nz, nghost_z, dz);
            double kapdcdzbot0 = nu / prandtl *
                (avg[nghost_z] - avg[nghost_z - 1]) /
                (0.5 * (dz[nghost_z] + dz[nghost_z - 1]));
            double kapdcdztop0 = nu / prandtl *
                (avg[n2 - nghost_z] - avg[n2 - nghost_z - 1]) /
                (0.5 * (dz[n2 - nghost_z] + dz[n2 - nghost_z - 1]));
            double ccenterline0 = nz % 2 == 0 ? 0.5 * (avg[nghost_z + nz / 2 - 1] + avg[nghost_z + nz / 2])
                                                     : avg[nghost_z + nz / 2];

            for (i = 0; i < local_size; i++) h[i] = gamdt;
            solve_helmholtz1d_u2(h, h, local_nx, local_ny, nz,
                                 nghost_x, nghost_y, nghost_z, dz, -nubdt / prandtl,
                                 0.0, 0.0, bctype_c_bot, bctype_c_top);

            get_xy_avg(h, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            double cvolavg1 = get_z_avg(avg, nz, nghost_z, dz);
            double kapdcdzbot1 = nu / prandtl *
                (avg[nghost_z] - avg[nghost_z - 1]) /
                (0.5 * (dz[nghost_z] + dz[nghost_z - 1]));
            double kapdcdztop1 = nu / prandtl *
                (avg[n2 - nghost_z] - avg[n2 - nghost_z - 1]) /
                (0.5 * (dz[n2 - nghost_z] + dz[n2 - nghost_z - 1]));
            double ccenterline1 = nz % 2 == 0 ? 0.5 * (avg[nghost_z + nz / 2 - 1] + avg[nghost_z + nz / 2])
                                                     : avg[nghost_z + nz / 2];

            double dcdt = cvolavgmode == 1 ? (want_cvolavg - cvolavg0) / cvolavg1 : 0.0;

            cvolavg0 += cvolavg1 * dcdt;
            kapdcdzbot0 += kapdcdzbot1 * dcdt;
            kapdcdztop0 += kapdcdztop1 * dcdt;
            ccenterline0 += ccenterline1 * dcdt;
            for (i = 0; i < local_size; i++) {
                c[i] += h[i] * dcdt;
                fc[i] += dcdt;
            }

            if (step == 2)
                io_append_ascii( (double []) { (double) (it + 1),
                    dcdt, kapdcdzbot0, kapdcdztop0, cvolavg0, ccenterline0 },
                    6, "outputdir/cstat.dat");
            // End c statistics

            apply_bc_w2(w, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_w_bot, bc_w_top, bctype_w_bot, bctype_w_top);
            helmholtz_operator_w2(w, h, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt);
            for (i = 0; i < local_size; i++) {
                h[i] += gamdt * fw[i] + zetdt * gw[i];
            }
            for (i = 0; i < local_size; i++) bc[i] = 0.0;
            apply_bc_w2(bc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_w_bot, bc_w_top, bctype_w_bot, bctype_w_top);
            helmholtz_operator_w2(bc, hbc, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, -nubdt);
            for (i = 0; i < local_size; i++) h[i] -= hbc[i];

            FFT3D_FWD(h_3d, zwork_d1, zwork_c1);

            solve_helmholtz_w2(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt, 0.0, 0.0, bctype_w_bot, bctype_w_top);

            FFT3D_BWD(w_3d, zwork_d1, zwork_c1);

            ghostz_periodic(w, ghostz_p);
            apply_bc_w2(w, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_w_bot, bc_w_top, bctype_w_bot, bctype_w_top);

            divergence2(u, v, w, h, local_nx, local_ny, nz,
                        nghost_x, nghost_y, nghost_z, dx, dy, dz);

            FFT3D_FWD(h_3d, zwork_d1, zwork_c1);

            pressure_poisson2(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, abdt);

            FFT3D_BWD(p_3d, zwork_d1, zwork_c1);

            ghostz_periodic(p, ghostz_p);
            project2(u, v, w, p, local_nx, local_ny, nz,
                     nghost_x, nghost_y, nghost_z, dx, dy, dz, abdt);
        }

        if ((it + 1) % istat == 0) {
            get_xy_avg(u, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/u_avg_it%d.dat", it + 1);
            io_write_ascii(avg, n2, fname);
            get_xy_avg(v, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/v_avg_it%d.dat", it + 1);
            io_write_ascii(avg, n2, fname);
            get_xy_avg(w, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/w_avg_it%d.dat", it + 1);
            io_write_ascii(avg, n2, fname);
            get_xy_avg(c, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/c_avg_it%d.dat", it + 1);
            io_write_ascii(avg, n2, fname);
            get_xy_avg(p, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/p_avg_it%d.dat", it + 1);
            io_write_ascii(avg, n2, fname);
            get_xy_corr(u, u, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/uu_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(v, v, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/vv_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(w, w, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/ww_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(c, c, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/cc_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(p, p, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/pp_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(w, u, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/wu_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(u, c, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/uc_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(w, c, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/wc_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
/*
            // x-average
            get_x_avg(u, xavg,
                      local_nx, local_ny, nz, decomp_c->zst[1] - 1,
                      nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/u_xavg_it%d.dat", it + 1);
            io_write_ascii(xavg, xavg_size, fname);
            get_x_avg(w, xavg,
                      local_nx, local_ny, nz, decomp_c->zst[1] - 1,
                      nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/w_xavg_it%d.dat", it + 1);
            io_write_ascii(xavg, xavg_size, fname);

            // x-correlation
            get_x_corr(w, u, xcorr,
                       local_nx, local_ny, nz, decomp_c->zst[1] - 1,
                       nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/wu_xcorr_it%d.dat", it + 1);
            io_write_ascii(xcorr, xcorr_size, fname);

            // y-average
            get_y_avg(u, yavg,
                      local_nx, ny, nz, nghost_x, nghost_y, nghost_z);
            sprintf(fname, "outputdir/u_yavg_it%d.dat", it + 1);
            io_write_jslice(yavg, fname,
                            local_nx, 1, nz, nghost_x, 0, nghost_z, local_x_start, 0);
            get_y_avg(w, yavg,
                      local_nx, ny, nz, nghost_x, nghost_y, nghost_z);
            sprintf(fname, "outputdir/w_yavg_it%d.dat", it + 1);
            io_write_jslice(yavg, fname,
                            local_nx, 1, nz, nghost_x, 0, nghost_z, local_x_start, 0);

            // y-correlation
            get_y_corr(w, u, ycorr,
                       local_nx, ny, nz, nghost_x, nghost_y, nghost_z);
            sprintf(fname, "outputdir/wu_ycorr_it%d.dat", it + 1);
            io_write_jslice(ycorr, fname,
                            local_nx, 1, nz, nghost_x, 0, nghost_z, local_x_start, 0);
*/
            get_xy_moment(u, mom3, 3.0,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/u_mom3_it%d.dat", it + 1);
            io_write_ascii(mom3, n2, fname);
            get_xy_moment(u, mom4, 4.0,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/u_mom4_it%d.dat", it + 1);
            io_write_ascii(mom4, n2, fname);

            FFT3D_FWD(u_3d, zwork_d1, zwork_c1);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/uu_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/uu_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(v_3d, zwork_d1, zwork_c1);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/vv_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/vv_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(w_3d, zwork_d1, zwork_c1);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/ww_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/ww_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(c_3d, zwork_d1, zwork_c1);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/cc_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/cc_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(p_3d, zwork_d1, zwork_c1);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/pp_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/pp_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(w_3d, zwork_d1, zwork_c1);
            FFT3D_FWD(u_3d, zwork_d2, zwork_c2);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c2[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/wu_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/wu_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(u_3d, zwork_d1, zwork_c1);
            FFT3D_FWD(c_3d, zwork_d2, zwork_c2);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c2[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/uc_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/uc_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(w_3d, zwork_d1, zwork_c1);
            FFT3D_FWD(c_3d, zwork_d2, zwork_c2);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c2[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/wc_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/wc_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);

            io_write_ascii(zp, n2, "outputdir/zp.dat");
            io_write_ascii(zw, n2, "outputdir/zw.dat");

            sprintf(fname, "outputdir/u_it%d.dat", it + 1);
            ghostz_truncate_double(u_3d, zwork_d1, ghostz_p);
            io_write(zwork_d1, fname, nx, ny, nz,
                nghost_z, decomp_d->zst, decomp_d->zsz);
            sprintf(fname, "outputdir/v_it%d.dat", it + 1);
            ghostz_truncate_double(v_3d, zwork_d1, ghostz_p);
            io_write(zwork_d1, fname, nx, ny, nz,
                nghost_z, decomp_d->zst, decomp_d->zsz);
            sprintf(fname, "outputdir/w_it%d.dat", it + 1);
            ghostz_truncate_double(w_3d, zwork_d1, ghostz_p);
            io_write(zwork_d1, fname, nx, ny, nz,
                nghost_z, decomp_d->zst, decomp_d->zsz);
            sprintf(fname, "outputdir/c_it%d.dat", it + 1);
            ghostz_truncate_double(c_3d, zwork_d1, ghostz_p);
            io_write(zwork_d1, fname, nx, ny, nz,
                nghost_z, decomp_d->zst, decomp_d->zsz);
/*
            sprintf(fname, "c_it%d", it + 1);
            io_write_scalarfield_vtk(c, xp, yp, zp,
                                     0, 0, 0, n0, n1, n2, n0, n1, n2,
                                     "outputdir", fname);
*/
        }
    }

    free(avg);
    free(corr);
    free(mom3);
    free(mom4);
    free(specx);
    free(specy);
    free(xavg);
    free(xcorr);
    //free(yavg);
    //free(ycorr);

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

    free_d3tensor(bc_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(hbc_3d, zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(v_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(w_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(c_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(p_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(h_3d, zst_pad[0], zen_pad[0],
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
    free_d3tensor(fu_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(fv_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(fw_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(fc_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(gu_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(gv_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(gw_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(gc_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);

    free(xp);
    free(xu);
    free(yp);
    free(yv);
    free(dz);
    free(zp);
    free(zw);

    diffuseu_finalize2();
    diffusew_finalize2();
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


void channel4(int p_row, int p_col,
              int nx, int ny, int nz, int it, int nt, int istat,
              int nghost_x, int nghost_y, int nghost_z,
              double lx, double ly, double lz,
              double dt, double want_cfl, int cflmode,
              double nu, double prandtl,
              double dpdx, double want_uvolavg, int uvolavgmode,
              double want_cvolavg, int cvolavgmode,
              double setuframe, double setvframe,
              double betg_x, double betg_y, double betg_z)
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
    diffusew_initialize4(nz, nghost_z);
    pressure_initialize4(nz, nghost_z);

    int local_nx = decomp_d->zsz[0];
    int local_ny = decomp_d->zsz[1];
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    double dx = lx / (double) nx;
    double dy = ly / (double) ny;
    double *xp = (double *) malloc(sizeof(double) * n0);
    double *xu = (double *) malloc(sizeof(double) * n0);
    double *yp = (double *) malloc(sizeof(double) * n1);
    double *yv = (double *) malloc(sizeof(double) * n1);
    double *dz = (double *) malloc(sizeof(double) * n2);
    double *zp = (double *) malloc(sizeof(double) * n2);
    double *zw = (double *) malloc(sizeof(double) * n2);
    check(xp != NULL);
    check(xu != NULL);
    check(yp != NULL);
    check(yv != NULL);
    check(dz != NULL);
    check(zp != NULL);
    check(zw != NULL);

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
        dz[k] = lz * 0.5 * (cos(M_PI * (double) iz / (double) nz)
                          - cos(M_PI * (double) (iz + 1) / (double) nz));
        //dz[k] = lz / (double) nz;
    }
    ghost_even_z(dz, 1, 1, nz, 0, 0, nghost_z);
    get_z_at_p(dz, zp, nz, nghost_z);
    get_z_at_w(dz, zw, nz, nghost_z);

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

    double ***bc_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***hbc_3d = d3tensor(zst_pad[0], zen_pad[0],
                                zst_pad[1], zen_pad[1],
                                zst_pad[2], zen_pad[2]);
    double ***u_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***v_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***w_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***c_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***p_3d = d3tensor(zst_pad[0], zen_pad[0],
                              zst_pad[1], zen_pad[1],
                              zst_pad[2], zen_pad[2]);
    double ***h_3d = d3tensor(zst_pad[0], zen_pad[0],
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
    double ***fu_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***fv_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***fw_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***fc_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***gu_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***gv_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***gw_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);
    double ***gc_3d = d3tensor(zst_pad[0], zen_pad[0],
                               zst_pad[1], zen_pad[1],
                               zst_pad[2], zen_pad[2]);

    double *bc = &(bc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *hbc = &(hbc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *u = &(u_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *v = &(v_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *w = &(w_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *c = &(c_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *p = &(p_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *h = &(h_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *muphx = &(muphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mvphx = &(mvphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *mwphx = &(mwphx_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *muphy = muphx;
    double *mvphy = mvphx;
    double *mwphy = mwphx;
    double *muphz = muphx;
    double *mvphz = mvphx;
    double *mwphz = mwphx;
    double *mucphx = muphx;
    double *mvcphy = mvphx;
    double *mwcphz = mwphx;
    double *fu = &(fu_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fv = &(fv_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fw = &(fw_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *fc = &(fc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *gu = &(gu_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *gv = &(gv_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *gw = &(gw_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);
    double *gc = &(gc_3d[zst_pad[0]][zst_pad[1]][zst_pad[2]]);

    int specx_size = (nx / 2 + 1) * n2 * 2;
    int specy_size = (ny / 2 + 1) * n2 * 2;
    int xavg_size = ny * n2;
    int xcorr_size = ny * n2;
    //int yavg_size = n0 * n2;
    //int ycorr_size = n0 * n2;

    double *avg = (double *) malloc(sizeof(double) * n2);
    double *corr = (double *) malloc(sizeof(double) * n2);
    double *mom3 = (double *) malloc(sizeof(double) * n2);
    double *mom4 = (double *) malloc(sizeof(double) * n2);
    double *specx = (double *) malloc(sizeof(double) * specx_size);
    double *specy = (double *) malloc(sizeof(double) * specy_size);
    double *xavg = (double *) malloc(sizeof(double) * xavg_size);
    double *xcorr = (double *) malloc(sizeof(double) * xcorr_size);
    //double *yavg = (double *) malloc(sizeof(double) * yavg_size);
    //double *ycorr = (double *) malloc(sizeof(double) * ycorr_size);

    check(avg != NULL);
    check(corr != NULL);
    check(mom3 != NULL);
    check(mom4 != NULL);
    check(specx != NULL);
    check(specy != NULL);
    check(xavg != NULL);
    check(xcorr != NULL);
    //check(yavg != NULL);
    //check(ycorr != NULL);

    srand(nrank);

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (iz = 0; iz < nz; iz++) {
                i = local_ix + nghost_x;
                j = local_iy + nghost_y;
                k = iz + nghost_z;
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = 0.0;
                v[ijk] = 0.0;
                w[ijk] = 0.0;
                c[ijk] = 0.0;
            }

    for (i = 0; i < local_size; i++) hbc[i] = p[i] = h[i]
        = fu[i] = fv[i] = fw[i] = fc[i]
        = gu[i] = gv[i] = gw[i] = gc[i] = FILL_VALUE;

    char fname[100];

    sprintf(fname, "outputdir/u_it%d.dat", it);
    io_read(zwork_d1, fname, nx, ny, nz,
        nghost_z, decomp_d->zst, decomp_d->zsz);
    ghostz_pad_double(u_3d, zwork_d1, ghostz_p);
    sprintf(fname, "outputdir/v_it%d.dat", it);
    io_read(zwork_d1, fname, nx, ny, nz,
        nghost_z, decomp_d->zst, decomp_d->zsz);
    ghostz_pad_double(v_3d, zwork_d1, ghostz_p);
    sprintf(fname, "outputdir/w_it%d.dat", it);
    io_read(zwork_d1, fname, nx, ny, nz,
        nghost_z, decomp_d->zst, decomp_d->zsz);
    ghostz_pad_double(w_3d, zwork_d1, ghostz_p);
    sprintf(fname, "outputdir/c_it%d.dat", it);
    io_read(zwork_d1, fname, nx, ny, nz,
        nghost_z, decomp_d->zst, decomp_d->zsz);
    ghostz_pad_double(c_3d, zwork_d1, ghostz_p);

    uframe = setuframe;
    vframe = setvframe;
    double bctype_u_bot = DIRICHU;
    double bctype_u_top = DIRICHU;
    double bctype_v_bot = DIRICHU;
    double bctype_v_top = DIRICHU;
    double bctype_c_bot = DIRICHU;
    double bctype_c_top = DIRICHU;
    double bctype_w_bot = DIRICHW;
    double bctype_w_top = DIRICHW;

    int step;

    for ( ; it < nt; it++) {

        double cfl = get_cfl(u, v, w,
                             local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                             dx, dy, dz, dt);
        if (cflmode == 1) { // Make CFL = want_cfl
            dt *= cfl > 0.0 ? want_cfl / cfl : 1.0;
            cfl = cfl > 0.0 ? want_cfl       : cfl;
        }
        io_append_ascii((double []) { (double) it,
            dt, cfl }, 3, "outputdir/cfl.dat");

        for (step = 0; step < 3; step++) { // RK3 stepper

            {
                double *tu = fu; fu = gu; gu = tu; // Swap
                double *tv = fv; fv = gv; gv = tv;
                double *tw = fw; fw = gw; gw = tw;
                double *tc = fc; fc = gc; gc = tc;
            }

            ghostz_periodic(u, ghostz_p);
            apply_bc_u4(u, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xu, yp, dz, bc_u_bot, bc_u_top, bctype_u_bot, bctype_u_top);

            ghostz_periodic(v, ghostz_p);
            apply_bc_u4(v, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yv, dz, bc_v_bot, bc_v_top, bctype_v_bot, bctype_v_top);

            ghostz_periodic(c, ghostz_p);
            apply_bc_u4(c, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_c_bot, bc_c_top, bctype_c_bot, bctype_c_top);

            ghostz_periodic(w, ghostz_p);
            apply_bc_w4(w, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_w_bot, bc_w_top, bctype_w_bot, bctype_w_top);

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

            flux_quick(u, v, w, c, mucphx, mvcphy, mwcphz, local_nx, local_ny, nz,
                       nghost_x, nghost_y, nghost_z, dx, dy, dz);

            advect_scalar(mucphx, mvcphy, mwcphz, fc, local_nx, local_ny, nz,
                          nghost_x, nghost_y, nghost_z, dx, dy, dz);

            add_buoyancy4(c, fu, fv, fw, betg_x, betg_y, betg_z,
                          local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z);

            ghostz_periodic(fu, ghostz_p);
            ghostz_periodic(fv, ghostz_p);
            ghostz_periodic(fw, ghostz_p);
            ghostz_periodic(fc, ghostz_p);

            double nuadt = nu * rk3_alp[step] * dt;
            double nubdt = nu * rk3_bet[step] * dt;
            double abdt = (rk3_alp[step] + rk3_bet[step]) * dt;
            double gamdt = rk3_gam[step] * dt;
            double zetdt = rk3_zet[step] * dt;

            apply_bc_u4(u, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xu, yp, dz, bc_u_bot, bc_u_top, bctype_u_bot, bctype_u_top);
            helmholtz_operator_u4(u, h, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt);
            for (i = 0; i < local_size; i++) {
                h[i] += gamdt * fu[i] + zetdt * gu[i];
            }
            for (i = 0; i < local_size; i++) bc[i] = 0.0;
            apply_bc_u4(bc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xu, yp, dz, bc_u_bot, bc_u_top, bctype_u_bot, bctype_u_top);
            helmholtz_operator_u4(bc, hbc, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, -nubdt);
            for (i = 0; i < local_size; i++) h[i] -= hbc[i];

            FFT3D_FWD(h_3d, zwork_d1, zwork_c1);

            solve_helmholtz_u4(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt, 0.0, 0.0, bctype_u_bot, bctype_u_top);

            FFT3D_BWD(u_3d, zwork_d1, zwork_c1);

            ghostz_periodic(u, ghostz_p);
            apply_bc_u4(u, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xu, yp, dz, bc_u_bot, bc_u_top, bctype_u_bot, bctype_u_top);

            // Begin u statistics
            get_xy_avg(u, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            double uvolavg0 = get_z_avg(avg, nz, nghost_z, dz);
            double nududzbot0 = nu *
                (a1 * (avg[nghost_z] - avg[nghost_z - 1]) + a2 * (avg[nghost_z + 1] - avg[nghost_z - 2])) /
                (0.5 * (dz[nghost_z] + dz[nghost_z - 1]));
            double nududztop0 = nu *
                (a1 * (avg[n2 - nghost_z] - avg[n2 - nghost_z - 1]) + a2 * (avg[n2 - nghost_z + 1] - avg[n2 - nghost_z - 2])) /
                (0.5 * (dz[n2 - nghost_z] + dz[n2 - nghost_z - 1]));
            double ucenterline0 = nz % 2 == 0 ? 0.5 * (avg[nghost_z + nz / 2 - 1] + avg[nghost_z + nz / 2])
                                                     : avg[nghost_z + nz / 2];

            for (i = 0; i < local_size; i++) h[i] = gamdt;
            solve_helmholtz1d_u4(h, h, local_nx, local_ny, nz,
                                 nghost_x, nghost_y, nghost_z, dz, -nubdt,
                                 0.0, 0.0, bctype_u_bot, bctype_u_top);

            get_xy_avg(h, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            double uvolavg1 = get_z_avg(avg, nz, nghost_z, dz);
            double nududzbot1 = nu *
                (a1 * (avg[nghost_z] - avg[nghost_z - 1]) + a2 * (avg[nghost_z + 1] - avg[nghost_z - 2])) /
                (0.5 * (dz[nghost_z] + dz[nghost_z - 1]));
            double nududztop1 = nu *
                (a1 * (avg[n2 - nghost_z] - avg[n2 - nghost_z - 1]) + a2 * (avg[n2 - nghost_z + 1] - avg[n2 - nghost_z - 2])) /
                (0.5 * (dz[n2 - nghost_z] + dz[n2 - nghost_z - 1]));
            double ucenterline1 = nz % 2 == 0 ? 0.5 * (avg[nghost_z + nz / 2 - 1] + avg[nghost_z + nz / 2])
                                                     : avg[nghost_z + nz / 2];

            double mdpdx = uvolavgmode == 1 ? (want_uvolavg - uframe - uvolavg0) / uvolavg1 : -dpdx;

            uvolavg0 += uvolavg1 * mdpdx;
            nududzbot0 += nududzbot1 * mdpdx;
            nududztop0 += nududztop1 * mdpdx;
            ucenterline0 += ucenterline1 * mdpdx;
            for (i = 0; i < local_size; i++) {
                u[i] += h[i] * mdpdx;
                fu[i] += mdpdx;
            }

            if (step == 2)
                io_append_ascii( (double []) { (double) (it + 1),
                    mdpdx, nududzbot0, nududztop0, uvolavg0, ucenterline0 },
                    6, "outputdir/pstat.dat");
            // End u statistics

            apply_bc_u4(v, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yv, dz, bc_v_bot, bc_v_top, bctype_v_bot, bctype_v_top);
            helmholtz_operator_u4(v, h, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt);
            for (i = 0; i < local_size; i++) {
                h[i] += gamdt * fv[i] + zetdt * gv[i];
            }
            for (i = 0; i < local_size; i++) bc[i] = 0.0;
            apply_bc_u4(bc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yv, dz, bc_v_bot, bc_v_top, bctype_v_bot, bctype_v_top);
            helmholtz_operator_u4(bc, hbc, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, -nubdt);
            for (i = 0; i < local_size; i++) h[i] -= hbc[i];

            FFT3D_FWD(h_3d, zwork_d1, zwork_c1);

            solve_helmholtz_u4(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt, 0.0, 0.0, bctype_v_bot, bctype_v_top);

            FFT3D_BWD(v_3d, zwork_d1, zwork_c1);

            ghostz_periodic(v, ghostz_p);
            apply_bc_u4(v, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yv, dz, bc_v_bot, bc_v_top, bctype_v_bot, bctype_v_top);

            apply_bc_u4(c, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_c_bot, bc_c_top, bctype_c_bot, bctype_c_top);
            helmholtz_operator_u4(c, h, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt / prandtl);
            for (i = 0; i < local_size; i++) {
                h[i] += gamdt * fc[i] + zetdt * gc[i];
            }
            for (i = 0; i < local_size; i++) bc[i] = 0.0;
            apply_bc_u4(bc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_c_bot, bc_c_top, bctype_c_bot, bctype_c_top);
            helmholtz_operator_u4(bc, hbc, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, -nubdt / prandtl);
            for (i = 0; i < local_size; i++) h[i] -= hbc[i];

            FFT3D_FWD(h_3d, zwork_d1, zwork_c1);

            solve_helmholtz_u4(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt / prandtl, 0.0, 0.0, bctype_c_bot, bctype_c_top);

            FFT3D_BWD(c_3d, zwork_d1, zwork_c1);

            ghostz_periodic(c, ghostz_p);
            apply_bc_u4(c, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_c_bot, bc_c_top, bctype_c_bot, bctype_c_top);

            // Begin c statistics
            get_xy_avg(c, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            double cvolavg0 = get_z_avg(avg, nz, nghost_z, dz);
            double kapdcdzbot0 = nu / prandtl *
                (a1 * (avg[nghost_z] - avg[nghost_z - 1]) + a2 * (avg[nghost_z + 1] - avg[nghost_z - 2])) /
                (0.5 * (dz[nghost_z] + dz[nghost_z - 1]));
            double kapdcdztop0 = nu / prandtl *
                (a1 * (avg[n2 - nghost_z] - avg[n2 - nghost_z - 1]) + a2 * (avg[n2 - nghost_z + 1] - avg[n2 - nghost_z - 2])) /
                (0.5 * (dz[n2 - nghost_z] + dz[n2 - nghost_z - 1]));
            double ccenterline0 = nz % 2 == 0 ? 0.5 * (avg[nghost_z + nz / 2 - 1] + avg[nghost_z + nz / 2])
                                                     : avg[nghost_z + nz / 2];

            for (i = 0; i < local_size; i++) h[i] = gamdt;
            solve_helmholtz1d_u4(h, h, local_nx, local_ny, nz,
                                 nghost_x, nghost_y, nghost_z, dz, -nubdt / prandtl,
                                 0.0, 0.0, bctype_c_bot, bctype_c_top);

            get_xy_avg(h, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            double cvolavg1 = get_z_avg(avg, nz, nghost_z, dz);
            double kapdcdzbot1 = nu / prandtl *
                (a1 * (avg[nghost_z] - avg[nghost_z - 1]) + a2 * (avg[nghost_z + 1] - avg[nghost_z - 2])) /
                (0.5 * (dz[nghost_z] + dz[nghost_z - 1]));
            double kapdcdztop1 = nu / prandtl *
                (a1 * (avg[n2 - nghost_z] - avg[n2 - nghost_z - 1]) + a2 * (avg[n2 - nghost_z + 1] - avg[n2 - nghost_z - 2])) /
                (0.5 * (dz[n2 - nghost_z] + dz[n2 - nghost_z - 1]));
            double ccenterline1 = nz % 2 == 0 ? 0.5 * (avg[nghost_z + nz / 2 - 1] + avg[nghost_z + nz / 2])
                                                     : avg[nghost_z + nz / 2];

            double dcdt = cvolavgmode == 1 ? (want_cvolavg - cvolavg0) / cvolavg1 : 0.0;

            cvolavg0 += cvolavg1 * dcdt;
            kapdcdzbot0 += kapdcdzbot1 * dcdt;
            kapdcdztop0 += kapdcdztop1 * dcdt;
            ccenterline0 += ccenterline1 * dcdt;
            for (i = 0; i < local_size; i++) {
                c[i] += h[i] * dcdt;
                fc[i] += dcdt;
            }

            if (step == 2)
                io_append_ascii( (double []) { (double) (it + 1),
                    dcdt, kapdcdzbot0, kapdcdztop0, cvolavg0, ccenterline0 },
                    6, "outputdir/cstat.dat");
            // End c statistics

            apply_bc_w4(w, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_w_bot, bc_w_top, bctype_w_bot, bctype_w_top);
            helmholtz_operator_w4(w, h, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, nuadt);
            for (i = 0; i < local_size; i++) {
                h[i] += gamdt * fw[i] + zetdt * gw[i];
            }
            for (i = 0; i < local_size; i++) bc[i] = 0.0;
            apply_bc_w4(bc, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_w_bot, bc_w_top, bctype_w_bot, bctype_w_top);
            helmholtz_operator_w4(bc, hbc, local_nx, local_ny, nz,
                                  nghost_x, nghost_y, nghost_z, dx, dy, dz, -nubdt);
            for (i = 0; i < local_size; i++) h[i] -= hbc[i];

            FFT3D_FWD(h_3d, zwork_d1, zwork_c1);

            solve_helmholtz_w4(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, -nubdt, 0.0, 0.0, bctype_w_bot, bctype_w_top);

            FFT3D_BWD(w_3d, zwork_d1, zwork_c1);

            ghostz_periodic(w, ghostz_p);
            apply_bc_w4(w, local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z,
                        xp, yp, dz, bc_w_bot, bc_w_top, bctype_w_bot, bctype_w_top);

            divergence4(u, v, w, h, local_nx, local_ny, nz,
                        nghost_x, nghost_y, nghost_z, dx, dy, dz);

            FFT3D_FWD(h_3d, zwork_d1, zwork_c1);

            pressure_poisson4(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z,
                dx, dy, dz, abdt);

            FFT3D_BWD(p_3d, zwork_d1, zwork_c1);

            ghostz_periodic(p, ghostz_p);
            project4(u, v, w, p, local_nx, local_ny, nz,
                     nghost_x, nghost_y, nghost_z, dx, dy, dz, abdt);
        }

        if ((it + 1) % istat == 0) {
            get_xy_avg(u, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/u_avg_it%d.dat", it + 1);
            io_write_ascii(avg, n2, fname);
            get_xy_avg(v, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/v_avg_it%d.dat", it + 1);
            io_write_ascii(avg, n2, fname);
            get_xy_avg(w, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/w_avg_it%d.dat", it + 1);
            io_write_ascii(avg, n2, fname);
            get_xy_avg(c, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/c_avg_it%d.dat", it + 1);
            io_write_ascii(avg, n2, fname);
            get_xy_avg(p, avg,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/p_avg_it%d.dat", it + 1);
            io_write_ascii(avg, n2, fname);
            get_xy_corr(u, u, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/uu_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(v, v, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/vv_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(w, w, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/ww_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(c, c, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/cc_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(p, p, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/pp_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(w, u, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/wu_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(u, c, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/uc_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
            get_xy_corr(w, c, corr,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/wc_corr_it%d.dat", it + 1);
            io_write_ascii(corr, n2, fname);
/*
            // x-average
            get_x_avg(u, xavg,
                      local_nx, local_ny, nz, decomp_c->zst[1] - 1,
                      nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/u_xavg_it%d.dat", it + 1);
            io_write_ascii(xavg, xavg_size, fname);
            get_x_avg(w, xavg,
                      local_nx, local_ny, nz, decomp_c->zst[1] - 1,
                      nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/w_xavg_it%d.dat", it + 1);
            io_write_ascii(xavg, xavg_size, fname);

            // x-correlation
            get_x_corr(w, u, xcorr,
                       local_nx, local_ny, nz, decomp_c->zst[1] - 1,
                       nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/wu_xcorr_it%d.dat", it + 1);
            io_write_ascii(xcorr, xcorr_size, fname);

            // y-average
            get_y_avg(u, yavg,
                      local_nx, ny, nz, nghost_x, nghost_y, nghost_z);
            sprintf(fname, "outputdir/u_yavg_it%d.dat", it + 1);
            io_write_jslice(yavg, fname,
                            local_nx, 1, nz, nghost_x, 0, nghost_z, local_x_start, 0);
            get_y_avg(w, yavg,
                      local_nx, ny, nz, nghost_x, nghost_y, nghost_z);
            sprintf(fname, "outputdir/w_yavg_it%d.dat", it + 1);
            io_write_jslice(yavg, fname,
                            local_nx, 1, nz, nghost_x, 0, nghost_z, local_x_start, 0);

            // y-correlation
            get_y_corr(w, u, ycorr,
                       local_nx, ny, nz, nghost_x, nghost_y, nghost_z);
            sprintf(fname, "outputdir/wu_ycorr_it%d.dat", it + 1);
            io_write_jslice(ycorr, fname,
                            local_nx, 1, nz, nghost_x, 0, nghost_z, local_x_start, 0);
*/
            get_xy_moment(u, mom3, 3.0,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/u_mom3_it%d.dat", it + 1);
            io_write_ascii(mom3, n2, fname);
            get_xy_moment(u, mom4, 4.0,
                local_nx, local_ny, nz, nghost_x, nghost_y, nghost_z, nx, ny);
            sprintf(fname, "outputdir/u_mom4_it%d.dat", it + 1);
            io_write_ascii(mom4, n2, fname);

            FFT3D_FWD(u_3d, zwork_d1, zwork_c1);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/uu_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/uu_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(v_3d, zwork_d1, zwork_c1);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/vv_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/vv_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(w_3d, zwork_d1, zwork_c1);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/ww_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/ww_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(c_3d, zwork_d1, zwork_c1);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/cc_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/cc_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(p_3d, zwork_d1, zwork_c1);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/pp_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/pp_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(w_3d, zwork_d1, zwork_c1);
            FFT3D_FWD(u_3d, zwork_d2, zwork_c2);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c2[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/wu_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/wu_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(u_3d, zwork_d1, zwork_c1);
            FFT3D_FWD(c_3d, zwork_d2, zwork_c2);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c2[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/uc_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/uc_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);
            FFT3D_FWD(w_3d, zwork_d1, zwork_c1);
            FFT3D_FWD(c_3d, zwork_d2, zwork_c2);
            get_xy_spec(
                &(zwork_c1[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                &(zwork_c2[decomp_c->zst[0]][decomp_c->zst[1]][decomp_c->zst[2]]),
                specx, specy,
                nx, decomp_c->zsz[0], decomp_c->zst[0] - 1,
                ny, decomp_c->zsz[1], decomp_c->zst[1] - 1,
                nz, nghost_z);
            sprintf(fname, "outputdir/wc_specx_it%d.dat", it + 1);
            io_write_ascii(specx, specx_size, fname);
            sprintf(fname, "outputdir/wc_specy_it%d.dat", it + 1);
            io_write_ascii(specy, specy_size, fname);

            io_write_ascii(zp, n2, "outputdir/zp.dat");
            io_write_ascii(zw, n2, "outputdir/zw.dat");

            sprintf(fname, "outputdir/u_it%d.dat", it + 1);
            ghostz_truncate_double(u_3d, zwork_d1, ghostz_p);
            io_write(zwork_d1, fname, nx, ny, nz,
                nghost_z, decomp_d->zst, decomp_d->zsz);
            sprintf(fname, "outputdir/v_it%d.dat", it + 1);
            ghostz_truncate_double(v_3d, zwork_d1, ghostz_p);
            io_write(zwork_d1, fname, nx, ny, nz,
                nghost_z, decomp_d->zst, decomp_d->zsz);
            sprintf(fname, "outputdir/w_it%d.dat", it + 1);
            ghostz_truncate_double(w_3d, zwork_d1, ghostz_p);
            io_write(zwork_d1, fname, nx, ny, nz,
                nghost_z, decomp_d->zst, decomp_d->zsz);
            sprintf(fname, "outputdir/c_it%d.dat", it + 1);
            ghostz_truncate_double(c_3d, zwork_d1, ghostz_p);
            io_write(zwork_d1, fname, nx, ny, nz,
                nghost_z, decomp_d->zst, decomp_d->zsz);
/*
            sprintf(fname, "c_it%d", it + 1);
            io_write_scalarfield_vtk(c, xp, yp, zp,
                                     0, 0, 0, n0, n1, n2, n0, n1, n2,
                                     "outputdir", fname);
*/
        }
    }

    free(avg);
    free(corr);
    free(mom3);
    free(mom4);
    free(specx);
    free(specy);
    free(xavg);
    free(xcorr);
    //free(yavg);
    //free(ycorr);

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

    free_d3tensor(bc_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(hbc_3d, zst_pad[0], zen_pad[0],
                          zst_pad[1], zen_pad[1],
                          zst_pad[2], zen_pad[2]);
    free_d3tensor(u_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(v_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(w_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(c_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(p_3d, zst_pad[0], zen_pad[0],
                        zst_pad[1], zen_pad[1],
                        zst_pad[2], zen_pad[2]);
    free_d3tensor(h_3d, zst_pad[0], zen_pad[0],
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
    free_d3tensor(fu_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(fv_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(fw_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(fc_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(gu_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(gv_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(gw_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);
    free_d3tensor(gc_3d, zst_pad[0], zen_pad[0],
                         zst_pad[1], zen_pad[1],
                         zst_pad[2], zen_pad[2]);

    free(xp);
    free(xu);
    free(yp);
    free(yv);
    free(dz);
    free(zp);
    free(zw);

    diffuseu_finalize4();
    diffusew_finalize4();
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
