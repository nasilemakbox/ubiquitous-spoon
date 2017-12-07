#include "check.h"
#include "get.h"


void get_xy_spec(fftw_complex uhat[], fftw_complex vhat[],
                 double specx[], double specy[],
                 int nx, int local_nx, int local_x_start,
                 int ny, int local_ny, int local_y_start,
                 int nz, int nghost_z)
{
    int n2 = nz + 2 * nghost_z;
    int specx_size = (nx / 2 + 1) * n2 * 2;
    int specy_size = (ny / 2 + 1) * n2 * 2;

    {
        int i;
        for (i = 0; i < specx_size; i++) specx[i] = 0.0;
        for (i = 0; i < specy_size; i++) specy[i] = 0.0;
    }

    int local_ix, local_iy, k;

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++)
            for (k = 0; k < n2; k++) {
                int ixiyk = (local_ix * local_ny + local_iy) * n2 + k;
                double re_uhat = uhat[ixiyk][0];
                double im_uhat = uhat[ixiyk][1];
                double re_vhat = vhat[ixiyk][0];
                double im_vhat = vhat[ixiyk][1];
                double re_spec = re_uhat * re_vhat + im_uhat * im_vhat;
                double im_spec = im_uhat * re_vhat - re_uhat * im_vhat;

                int ix = local_ix + local_x_start;
                int iy = local_iy + local_y_start;

                int ixk2 = (ix * n2 + k) * 2;
                int iyk2 = (iy * n2 + k) * 2;
                int niy = (ny - iy) % ny;
                int niyk2 = (niy * n2 + k) * 2;

                // specx
                if (ix == 0 && iy == 0) {
                }
                else {
                    specx[ixk2] += re_spec;
                    specx[ixk2 + 1] += im_spec;
                }

                // specy
                if (ix == 0) {
                    if (0 < iy && iy <= ny / 2) {
                        specy[iyk2] += re_spec;
                        specy[iyk2 + 1] += im_spec;
                    }
                }
                else if (ix == nx / 2) {
                    if (nx % 2 == 0) { // nx even
                        if (0 <= iy && iy <= ny / 2) {
                            specy[iyk2] += re_spec;
                            specy[iyk2 + 1] += im_spec;
                        }
                    }
                    else { // nx odd
                        if (0 <= iy && iy <= ny / 2) {
                            specy[iyk2] += re_spec;
                            specy[iyk2 + 1] += im_spec;
                        }
                        if (0 <= niy && niy <= ny / 2) {
                            specy[niyk2] += re_spec;
                            specy[niyk2 + 1] += -im_spec;
                        }
                    }
                }
                else {
                    if (0 <= iy && iy <= ny / 2) {
                        specy[iyk2] += re_spec;
                        specy[iyk2 + 1] += im_spec;
                    }
                    if (0 <= niy && niy <= ny / 2) {
                        specy[niyk2] += re_spec;
                        specy[niyk2 + 1] += -im_spec;
                    }
                }
            }

    MPI_Allreduce(MPI_IN_PLACE, specx, specx_size,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, specy, specy_size,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}


void get_xy_moment(double u[], double moment[], double en,
                   int local_nx, int local_ny, int nz,
                   int nghost_x, int nghost_y, int nghost_z,
                   int nx, int ny)
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    double *u_avg = (double *) malloc(sizeof(double) * n2);

    check(u_avg != NULL);

    get_xy_avg(u, u_avg, local_nx, local_ny, nz,
        nghost_x, nghost_y, nghost_z, nx, ny);

    for (k = 0; k < n2; k++) moment[k] = 0.0;

    for (i = nghost_x; i < n0 - nghost_x; i++)
        for (j = nghost_y; j < n1 - nghost_y; j++)
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                moment[k] += pow(u[ijk] - u_avg[k], en);
            }

    MPI_Allreduce(MPI_IN_PLACE, moment, n2,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double scale = 1.0 / ((double) nx * (double) ny);

    for (k = 0; k < n2; k++) moment[k] *= scale;

    free(u_avg);
}


void get_xy_corr(double u[], double v[], double corr[],
                 int local_nx, int local_ny, int nz,
                 int nghost_x, int nghost_y, int nghost_z,
                 int nx, int ny)
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    double *u_avg = (double *) malloc(sizeof(double) * n2);
    double *v_avg = (double *) malloc(sizeof(double) * n2);

    check(u_avg != NULL);
    check(v_avg != NULL);

    get_xy_avg(u, u_avg, local_nx, local_ny, nz,
        nghost_x, nghost_y, nghost_z, nx, ny);
    get_xy_avg(v, v_avg, local_nx, local_ny, nz,
        nghost_x, nghost_y, nghost_z, nx, ny);

    for (k = 0; k < n2; k++) corr[k] = 0.0;

    for (i = nghost_x; i < n0 - nghost_x; i++)
        for (j = nghost_y; j < n1 - nghost_y; j++)
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                corr[k] += (u[ijk] - u_avg[k]) * (v[ijk] - v_avg[k]);
            }

    MPI_Allreduce(MPI_IN_PLACE, corr, n2,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double scale = 1.0 / ((double) nx * (double) ny);

    for (k = 0; k < n2; k++) corr[k] *= scale;

    free(u_avg);
    free(v_avg);
}


/*
void get_y_corr(double u[], double v[], double corr[],
                int local_nx, int ny, int nz,
                int nghost_x, int nghost_y, int nghost_z)
// Get y-correlation for every location in the xz-plane.
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    double *u_avg = (double *) malloc(sizeof(double) * n0 * n2);
    double *v_avg = (double *) malloc(sizeof(double) * n0 * n2);

    check(u_avg != NULL);
    check(v_avg != NULL);

    get_y_avg(u, u_avg, local_nx, ny, nz, nghost_x, nghost_y, nghost_z);
    get_y_avg(v, v_avg, local_nx, ny, nz, nghost_x, nghost_y, nghost_z);

    for (i = 0; i < n0; i++)
        for (k = 0; k < n2; k++) {
            int ik = i * n2 + k;
            corr[ik] = 0.0;
        }

    for (i = 0; i < n0; i++)
        for (j = nghost_y; j < n1 - nghost_y; j++)
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int ik = i * n2 + k;
                corr[ik] += (u[ijk] - u_avg[ik]) * (v[ijk] - v_avg[ik]);
            }

    double scale = 1.0 / (double) ny;

    for (i = 0; i < n0; i++)
        for (k = 0; k < n2; k++) {
            int ik = i * n2 + k;
            corr[ik] *= scale;
        }

    free(u_avg);
    free(v_avg);
}
*/


void get_x_corr(double u[], double v[], double corr[],
                int local_nx, int local_ny, int nz,
                int local_y_start,
                int nghost_x, int nghost_y, int nghost_z,
                int nx, int ny)
// Get x-correlation for every location in the yz-plane.
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;
    int iy;

    double *u_avg = (double *) malloc(sizeof(double) * ny * n2);
    double *v_avg = (double *) malloc(sizeof(double) * ny * n2);

    check(u_avg != NULL);
    check(v_avg != NULL);

    get_x_avg(u, u_avg, local_nx, local_ny, nz, local_y_start,
        nghost_x, nghost_y, nghost_z, nx, ny);
    get_x_avg(v, v_avg, local_nx, local_ny, nz, local_y_start,
        nghost_x, nghost_y, nghost_z, nx, ny);

    for (iy = 0; iy < ny; iy++)
        for (k = 0; k < n2; k++) {
            int iyk = iy * n2 + k;
            corr[iyk] = 0.0;
        }

    for (i = nghost_x; i < n0 - nghost_x; i++)
        for (j = nghost_y; j < n1 - nghost_y; j++)
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int local_iy = j - nghost_y;
                iy = local_y_start + local_iy;
                int iyk = iy * n2 + k;
                corr[iyk] += (u[ijk] - u_avg[iyk]) * (v[ijk] - v_avg[iyk]);
            }

    MPI_Allreduce(MPI_IN_PLACE, corr, ny * n2,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double scale = 1.0 / (double) nx;

    for (iy = 0; iy < ny; iy++)
        for (k = 0; k < n2; k++) {
            int iyk = iy * n2 + k;
            corr[iyk] *= scale;
        }

    free(u_avg);
    free(v_avg);
}


void get_xy_avg(double u[], double avg[],
                int local_nx, int local_ny, int nz,
                int nghost_x, int nghost_y, int nghost_z,
                int nx, int ny)
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (k = 0; k < n2; k++) avg[k] = 0.0;

    for (i = nghost_x; i < n0 - nghost_x; i++)
        for (j = nghost_y; j < n1 - nghost_y; j++)
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                avg[k] += u[ijk];
            }

    MPI_Allreduce(MPI_IN_PLACE, avg, n2,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double scale = 1.0 / ((double) nx * (double) ny);

    for (k = 0; k < n2; k++) avg[k] *= scale;
}


/*
void get_y_avg(double u[], double avg[],
               int local_nx, int ny, int nz,
               int nghost_x, int nghost_y, int nghost_z)
// Get y-average for every location in the xz-plane.
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0; i++)
        for (k = 0; k < n2; k++) {
            int ik = i * n2 + k;
            avg[ik] = 0.0; // Initialize to zero.
        }

    for (i = 0; i < n0; i++)
        for (j = nghost_y; j < n1 - nghost_y; j++)
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int ik = i * n2 + k;
                avg[ik] += u[ijk]; // Sum
            }

    double scale = 1.0 / (double) ny;

    for (i = 0; i < n0; i++)
        for (k = 0; k < n2; k++) {
            int ik = i * n2 + k;
            avg[ik] *= scale;
        }
}
*/


void get_x_avg(double u[], double avg[],
               int local_nx, int local_ny, int nz,
               int local_y_start,
               int nghost_x, int nghost_y, int nghost_z,
               int nx, int ny)
// Get x-average for every location in the yz-plane.
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;
    int iy;

    for (iy = 0; iy < ny; iy++)
        for (k = 0; k < n2; k++) {
            int iyk = iy * n2 + k;
            avg[iyk] = 0.0; // Initialize to zero.
        }

    for (i = nghost_x; i < n0 - nghost_x; i++)
        for (j = nghost_y; j < n1 - nghost_y; j++)
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int local_iy = j - nghost_y;
                iy = local_y_start + local_iy;
                int iyk = iy * n2 + k;
                avg[iyk] += u[ijk]; // Sum
            }

    MPI_Allreduce(MPI_IN_PLACE, avg, ny * n2,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double scale = 1.0 / (double) nx;

    for (iy = 0; iy < ny; iy++)
        for (k = 0; k < n2; k++) {
            int iyk = iy * n2 + k;
            avg[iyk] *= scale;
        }
}


double get_cfl(double u[], double v[], double w[],
               int local_nx, int local_ny, int nz,
               int nghost_x, int nghost_y, int nghost_z,
               double dx, double dy, double dz[], double dt)
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    double c, cfl = 0.0;

    for (i = nghost_x; i < n0 - nghost_x; i++)
        for (j = nghost_y; j < n1 - nghost_y; j++)
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                double dzph = 0.5 * (dz[k] + dz[k + 1]);
                if (cfl < (c = fabs(u[ijk]) * dt / dx)) cfl = c;
                if (cfl < (c = fabs(v[ijk]) * dt / dy)) cfl = c;
                if (cfl < (c = fabs(w[ijk]) * dt / dzph)) cfl = c;
            }

    MPI_Allreduce(MPI_IN_PLACE, &cfl, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return cfl;
}


void get_avg_blowing_z(double u[],
                       int local_nx, int local_ny, int nz,
                       int nghost_x, int nghost_y, int nghost_z,
                       int nx, int ny, double *avg_bot, double *avg_top)
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    *avg_bot = 0.0;
    *avg_top = 0.0;

    for (i = nghost_x; i < n0 - nghost_x; i++)
        for (j = nghost_y; j < n1 - nghost_y; j++) {
            k = nghost_z - 1; {
                int ijk = (i * n1 + j) * n2 + k;
                *avg_bot += u[ijk];
            }
            k = n2 - nghost_z - 1; {
                int ijk = (i * n1 + j) * n2 + k;
                *avg_top -= u[ijk];
            }
        }

    MPI_Allreduce(MPI_IN_PLACE, avg_bot, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, avg_top, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double scale = 1.0 / ((double) nx * (double) ny);

    *avg_bot *= scale;
    *avg_top *= scale;
}


void get_z_at_p(double dz[], double z[], int nz, int nghost_z)
{
    int n2 = nz + 2 * nghost_z;

    int k;

    z[0] = 0.5 * dz[0];

    for (k = 1; k < n2; k++) z[k] = z[k - 1] + 0.5 * (dz[k - 1] + dz[k]);

    double z0 = 0.5 * (z[nghost_z - 1] + z[nghost_z]);

    for (k = 0; k < n2; k++) z[k] -= z0;
}


void get_z_at_w(double dz[], double z[], int nz, int nghost_z)
{
    int n2 = nz + 2 * nghost_z;

    int k;

    z[0] = dz[0];

    for (k = 1; k < n2; k++) z[k] = z[k - 1] + dz[k];

    double z0 = z[nghost_z - 1];

    for (k = 0; k < n2; k++) z[k] -= z0;
}


double get_z_avg(double u[], int nz, int nghost_z, double dz[])
{
    int k;

    double avg = 0.0, lz = 0.0;

    for (k = nghost_z; k < nz + nghost_z; k++) {
        avg += u[k] * dz[k];
        lz += dz[k];
    }

    return avg / lz;
}
