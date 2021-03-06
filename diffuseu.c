#include "check.h"
#include "tridag.h"
#include "heptdag.h"
#include "fourth.h"
#include "diffuseu.h"


static double *a;
static double *b;
static double *c;
static double *d;
static double *e;
static double *f;
static double *g;
static double *r;
static double *x;

#define SECOND (2)
#define FOURTH (4)

static int order = 0;

static void ghost_stencil_bot2(double dz[], int k, double bc, int bctype);
static void ghost_stencil_top2(double dz[], int k, double bc, int bctype);
static void ghost_reduce_bot2(int k);
static void ghost_reduce_top2(int k);
static void ghost_fill_bot2(int k);
static void ghost_fill_top2(int k);
static void ghost_stencil_bot4(double dz[], int k, double bc, int bctype);
static void ghost_stencil_top4(double dz[], int k, double bc, int bctype);
static void ghost_reduce_bot4(int k);
static void ghost_reduce_top4(int k);
static void ghost_fill_bot4(int k);
static void ghost_fill_top4(int k);


void apply_bc_u2(double u[],
                 int local_nx, int local_ny, int nz,
                 int nghost_x, int nghost_y, int nghost_z,
                 double xu[], double yu[], double dz[],
                 double (*bc_bot)(double, double),
                 double (*bc_top)(double, double),
                 int bctype_bot, int bctype_top)
{
    check(order == SECOND);

    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    // Fill ghost cells
    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++) {
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                x[k] = u[ijk];
            }
            ghost_stencil_bot2(dz, nghost_z - 1,  bc_bot(xu[i], yu[j]), bctype_bot);
            ghost_stencil_top2(dz, n2 - nghost_z, bc_top(xu[i], yu[j]), bctype_top);
            ghost_fill_bot2(nghost_z - 1);
            ghost_fill_top2(n2 - nghost_z);
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = x[k];
            }
        }
}


void apply_bc_u4(double u[],
                 int local_nx, int local_ny, int nz,
                 int nghost_x, int nghost_y, int nghost_z,
                 double xu[], double yu[], double dz[],
                 double (*bc_bot)(double, double),
                 double (*bc_top)(double, double),
                 int bctype_bot, int bctype_top)
{
    check(order == FOURTH);

    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    // Fill ghost cells
    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++) {
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                x[k] = u[ijk];
            }
            ghost_stencil_bot4(dz, nghost_z - 1,  bc_bot(xu[i], yu[j]), bctype_bot);
            ghost_stencil_top4(dz, n2 - nghost_z, bc_top(xu[i], yu[j]), bctype_top);
            ghost_fill_bot4(nghost_z - 1);
            ghost_fill_top4(n2 - nghost_z);
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = x[k];
            }
        }
}


void helmholtz1d_operator_u2(double u[], double h[],
                             int local_nx, int local_ny, int nz,
                             int nghost_x, int nghost_y, int nghost_z,
                             double dz[], double nudt)
{
    check(order == SECOND);

    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++)
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                double dzp1h = 0.5 * (dz[k + 1] + dz[k]);
                double dzm1h = 0.5 * (dz[k - 1] + dz[k]);
                h[ijk] = u[ijk] + nudt * (
                               ((u[kp1] - u[ijk]) / dzp1h
                              - (u[ijk] - u[km1]) / dzm1h) / dz[k]);
            }
}


void helmholtz1d_operator_u4(double u[], double h[],
                             int local_nx, int local_ny, int nz,
                             int nghost_x, int nghost_y, int nghost_z,
                             double dz[], double nudt)
{
    check(order == FOURTH);

    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++)
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int km2 = (i * n1 + j) * n2 + (k - 2);
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                int km3 = (i * n1 + j) * n2 + (k - 3);
                int kp3 = (i * n1 + j) * n2 + (k + 3);
                double wp3h = a1 * (u[kp2] - u[kp1]) + a2 * (u[kp3] - u[ijk]);
                double wp1h = a1 * (u[kp1] - u[ijk]) + a2 * (u[kp2] - u[km1]);
                double wm1h = a1 * (u[ijk] - u[km1]) + a2 * (u[kp1] - u[km2]);
                double wm3h = a1 * (u[km1] - u[km2]) + a2 * (u[ijk] - u[km3]);
                double dzp3h = 0.5 * (dz[k + 2] + dz[k + 1]);
                double dzp1h = 0.5 * (dz[k + 1] + dz[k]);
                double dzm1h = 0.5 * (dz[k - 1] + dz[k]);
                double dzm3h = 0.5 * (dz[k - 2] + dz[k - 1]);
                h[ijk] = u[ijk] + nudt * (
                               (a1 * (wp1h / dzp1h - wm1h / dzm1h)
                              + a2 * (wp3h / dzp3h - wm3h / dzm3h)) / dz[k]);
            }
}


void solve_helmholtz1d_u2(double u[], double h[],
                          int local_nx, int local_ny, int nz,
                          int nghost_x, int nghost_y, int nghost_z,
                          double dz[], double nudt,
                          double avg_bot, double avg_top,
                          int bctype_bot, int bctype_top)
{
    check(order == SECOND);

    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (k = 0; k < n2; k++) x[k] = 0.0;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++) {
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                double dzp1h = 0.5 * (dz[k + 1] + dz[k]);
                double dzm1h = 0.5 * (dz[k - 1] + dz[k]);
                double dvol = dz[k];
                a[k] = + dvol * nudt / (dzm1h * dz[k]);
                c[k] = + dvol * nudt / (dzp1h * dz[k]);
                b[k] = - dvol * nudt / (dzm1h * dz[k])
                       - dvol * nudt / (dzp1h * dz[k])
                       + dvol;
                int ijk = (i * n1 + j) * n2 + k;
                r[k] = h[ijk] * dvol;
            }
            ghost_stencil_bot2(dz, nghost_z - 1,  avg_bot, bctype_bot);
            ghost_stencil_top2(dz, n2 - nghost_z, avg_top, bctype_top);
            ghost_reduce_bot2(nghost_z - 1);
            ghost_reduce_top2(n2 - nghost_z);
            {
                k = nghost_z;
                tridag(&a[k], &b[k], &c[k], &r[k], &x[k], nz);
            }
            ghost_fill_bot2(nghost_z - 1);
            ghost_fill_top2(n2 - nghost_z);
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = x[k];
            }
        }
}


void solve_helmholtz1d_u4(double u[], double h[],
                          int local_nx, int local_ny, int nz,
                          int nghost_x, int nghost_y, int nghost_z,
                          double dz[], double nudt,
                          double avg_bot, double avg_top,
                          int bctype_bot, int bctype_top)
{
    check(order == FOURTH);

    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (k = 0; k < n2; k++) x[k] = 0.0;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++) {
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                double dzp3h = 0.5 * (dz[k + 2] + dz[k + 1]);
                double dzp1h = 0.5 * (dz[k + 1] + dz[k]);
                double dzm1h = 0.5 * (dz[k - 1] + dz[k]);
                double dzm3h = 0.5 * (dz[k - 2] + dz[k - 1]);
                double dvol = dz[k];
                a[k] = + a2 * a2 * dvol * nudt / (dzm3h * dz[k]);
                g[k] = + a2 * a2 * dvol * nudt / (dzp3h * dz[k]);
                b[k] = + a1 * a2 * dvol * nudt / (dzm1h * dz[k])
                       + a1 * a2 * dvol * nudt / (dzm3h * dz[k]);
                f[k] = + a1 * a2 * dvol * nudt / (dzp1h * dz[k])
                       + a1 * a2 * dvol * nudt / (dzp3h * dz[k]);
                c[k] = - a1 * a2 * dvol * nudt / (dzp1h * dz[k])
                       - a1 * a2 * dvol * nudt / (dzm3h * dz[k])
                       + a1 * a1 * dvol * nudt / (dzm1h * dz[k]);
                e[k] = - a1 * a2 * dvol * nudt / (dzm1h * dz[k])
                       - a1 * a2 * dvol * nudt / (dzp3h * dz[k])
                       + a1 * a1 * dvol * nudt / (dzp1h * dz[k]);
                d[k] = - a2 * a2 * dvol * nudt / (dzm3h * dz[k])
                       - a2 * a2 * dvol * nudt / (dzp3h * dz[k])
                       - a1 * a1 * dvol * nudt / (dzm1h * dz[k])
                       - a1 * a1 * dvol * nudt / (dzp1h * dz[k])
                                 + dvol;
                int ijk = (i * n1 + j) * n2 + k;
                r[k] = h[ijk] * dvol;
            }
            ghost_stencil_bot4(dz, nghost_z - 1,  avg_bot, bctype_bot);
            ghost_stencil_top4(dz, n2 - nghost_z, avg_top, bctype_top);
            ghost_reduce_bot4(nghost_z - 1);
            ghost_reduce_top4(n2 - nghost_z);
            {
                k = nghost_z;
                heptdag(&a[k], &b[k], &c[k], &d[k], &e[k],
                        &f[k], &g[k], &r[k], &x[k], nz);
            }
            ghost_fill_bot4(nghost_z - 1);
            ghost_fill_top4(n2 - nghost_z);
            for (k = 0; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = x[k];
            }
        }
}


void add_laplacexy_operator_u2(double u[], double h[],
                               int local_nx, int local_ny, int nz,
                               int nghost_x, int nghost_y, int nghost_z,
                               double dx, double dy, double nu)
{
    check(order == SECOND);

    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 1; i < n0 - 1; i++)
        for (j = 1; j < n1 - 1; j++)
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                h[ijk] += nu * (
                               ((u[ip1] - u[ijk]) / dx
                              - (u[ijk] - u[im1]) / dx) / dx
                             + ((u[jp1] - u[ijk]) / dy
                              - (u[ijk] - u[jm1]) / dy) / dy);
            }
}


void add_laplacexy_operator_u4(double u[], double h[],
                               int local_nx, int local_ny, int nz,
                               int nghost_x, int nghost_y, int nghost_z,
                               double dx, double dy, double nu)
{
    check(order == FOURTH);

    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 3; i < n0 - 3; i++)
        for (j = 3; j < n1 - 3; j++)
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                int im2 = ((i - 2) * n1 + j) * n2 + k;
                int jm2 = (i * n1 + (j - 2)) * n2 + k;
                int ip2 = ((i + 2) * n1 + j) * n2 + k;
                int jp2 = (i * n1 + (j + 2)) * n2 + k;
                int im3 = ((i - 3) * n1 + j) * n2 + k;
                int jm3 = (i * n1 + (j - 3)) * n2 + k;
                int ip3 = ((i + 3) * n1 + j) * n2 + k;
                int jp3 = (i * n1 + (j + 3)) * n2 + k;
                double up3h = a1 * (u[ip2] - u[ip1]) + a2 * (u[ip3] - u[ijk]);
                double up1h = a1 * (u[ip1] - u[ijk]) + a2 * (u[ip2] - u[im1]);
                double um1h = a1 * (u[ijk] - u[im1]) + a2 * (u[ip1] - u[im2]);
                double um3h = a1 * (u[im1] - u[im2]) + a2 * (u[ijk] - u[im3]);
                double vp3h = a1 * (u[jp2] - u[jp1]) + a2 * (u[jp3] - u[ijk]);
                double vp1h = a1 * (u[jp1] - u[ijk]) + a2 * (u[jp2] - u[jm1]);
                double vm1h = a1 * (u[ijk] - u[jm1]) + a2 * (u[jp1] - u[jm2]);
                double vm3h = a1 * (u[jm1] - u[jm2]) + a2 * (u[ijk] - u[jm3]);
                h[ijk] += nu * (
                               (a1 * (up1h / dx    - um1h / dx   )
                              + a2 * (up3h / dx    - um3h / dx   )) / dx
                             + (a1 * (vp1h / dy    - vm1h / dy   )
                              + a2 * (vp3h / dy    - vm3h / dy   )) / dy);
            }
}


void helmholtz_operator_u2(double u[], double h[],
                           int local_nx, int local_ny, int nz,
                           int nghost_x, int nghost_y, int nghost_z,
                           double dx, double dy, double dz[], double nudt)
{
    check(order == SECOND);

    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 1; i < n0 - 1; i++)
        for (j = 1; j < n1 - 1; j++)
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                double dzp1h = 0.5 * (dz[k + 1] + dz[k]);
                double dzm1h = 0.5 * (dz[k - 1] + dz[k]);
                h[ijk] = u[ijk] + nudt * (
                               ((u[ip1] - u[ijk]) / dx
                              - (u[ijk] - u[im1]) / dx) / dx
                             + ((u[jp1] - u[ijk]) / dy
                              - (u[ijk] - u[jm1]) / dy) / dy
                             + ((u[kp1] - u[ijk]) / dzp1h
                              - (u[ijk] - u[km1]) / dzm1h) / dz[k]);
            }
}


void helmholtz_operator_u4(double u[], double h[],
                           int local_nx, int local_ny, int nz,
                           int nghost_x, int nghost_y, int nghost_z,
                           double dx, double dy, double dz[], double nudt)
{
    check(order == FOURTH);

    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 3; i < n0 - 3; i++)
        for (j = 3; j < n1 - 3; j++)
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int im2 = ((i - 2) * n1 + j) * n2 + k;
                int jm2 = (i * n1 + (j - 2)) * n2 + k;
                int km2 = (i * n1 + j) * n2 + (k - 2);
                int ip2 = ((i + 2) * n1 + j) * n2 + k;
                int jp2 = (i * n1 + (j + 2)) * n2 + k;
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                int im3 = ((i - 3) * n1 + j) * n2 + k;
                int jm3 = (i * n1 + (j - 3)) * n2 + k;
                int km3 = (i * n1 + j) * n2 + (k - 3);
                int ip3 = ((i + 3) * n1 + j) * n2 + k;
                int jp3 = (i * n1 + (j + 3)) * n2 + k;
                int kp3 = (i * n1 + j) * n2 + (k + 3);
                double up3h = a1 * (u[ip2] - u[ip1]) + a2 * (u[ip3] - u[ijk]);
                double up1h = a1 * (u[ip1] - u[ijk]) + a2 * (u[ip2] - u[im1]);
                double um1h = a1 * (u[ijk] - u[im1]) + a2 * (u[ip1] - u[im2]);
                double um3h = a1 * (u[im1] - u[im2]) + a2 * (u[ijk] - u[im3]);
                double vp3h = a1 * (u[jp2] - u[jp1]) + a2 * (u[jp3] - u[ijk]);
                double vp1h = a1 * (u[jp1] - u[ijk]) + a2 * (u[jp2] - u[jm1]);
                double vm1h = a1 * (u[ijk] - u[jm1]) + a2 * (u[jp1] - u[jm2]);
                double vm3h = a1 * (u[jm1] - u[jm2]) + a2 * (u[ijk] - u[jm3]);
                double wp3h = a1 * (u[kp2] - u[kp1]) + a2 * (u[kp3] - u[ijk]);
                double wp1h = a1 * (u[kp1] - u[ijk]) + a2 * (u[kp2] - u[km1]);
                double wm1h = a1 * (u[ijk] - u[km1]) + a2 * (u[kp1] - u[km2]);
                double wm3h = a1 * (u[km1] - u[km2]) + a2 * (u[ijk] - u[km3]);
                double dzp3h = 0.5 * (dz[k + 2] + dz[k + 1]);
                double dzp1h = 0.5 * (dz[k + 1] + dz[k]);
                double dzm1h = 0.5 * (dz[k - 1] + dz[k]);
                double dzm3h = 0.5 * (dz[k - 2] + dz[k - 1]);
                h[ijk] = u[ijk] + nudt * (
                               (a1 * (up1h / dx    - um1h / dx   )
                              + a2 * (up3h / dx    - um3h / dx   )) / dx
                             + (a1 * (vp1h / dy    - vm1h / dy   )
                              + a2 * (vp3h / dy    - vm3h / dy   )) / dy
                             + (a1 * (wp1h / dzp1h - wm1h / dzm1h)
                              + a2 * (wp3h / dzp3h - wm3h / dzm3h)) / dz[k]);
            }
}


void solve_helmholtz_u2(fftw_complex cout[],
                        int nx, int local_nx, int local_x_start,
                        int ny, int local_ny, int local_y_start,
                        int nz, int nghost_z,
                        double dx, double dy, double dz[], double nudt,
                        double avg_bot, double avg_top,
                        int bctype_bot, int bctype_top)
{
    check(order == SECOND);

    int n2 = nz + 2 * nghost_z;

    int local_ix, local_iy, rc, k;

    for (k = 0; k < n2; k++) x[k] = 0.0;

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++) {

            int ix = local_ix + local_x_start;
            int iy = local_iy + local_y_start;
            double cosx1 = cos(2.0 * M_PI * 1.0 * (double) ix / (double) nx);
            double cosy1 = cos(2.0 * M_PI * 1.0 * (double) iy / (double) ny);
            double lamx = 2.0 - 2.0 * cosx1;
            double lamy = 2.0 - 2.0 * cosy1;

            for (rc = 0; rc < 2; rc++) {
                for (k = nghost_z; k < n2 - nghost_z; k++) {
                    double dzp1h = 0.5 * (dz[k + 1] + dz[k]);
                    double dzm1h = 0.5 * (dz[k - 1] + dz[k]);
                    double dvol = dx * dy * dz[k];
                    a[k] =     + dvol * nudt / (dzm1h * dz[k]);
                    c[k] =     + dvol * nudt / (dzp1h * dz[k]);
                    b[k] =     - dvol * nudt / (dzm1h * dz[k])
                               - dvol * nudt / (dzp1h * dz[k])
                        - lamx * dvol * nudt / (dx    * dx)
                        - lamy * dvol * nudt / (dy    * dy)
                               + dvol;
                    int ixiyk = (local_ix * local_ny + local_iy) * n2 + k;
                    r[k] = cout[ixiyk][rc] * dvol;
                }
                if (ix == 0 && iy == 0 && rc == 0) {
                    ghost_stencil_bot2(dz, nghost_z - 1,  avg_bot, bctype_bot);
                    ghost_stencil_top2(dz, n2 - nghost_z, avg_top, bctype_top);
                }
                else {
                    ghost_stencil_bot2(dz, nghost_z - 1,  0.0, bctype_bot);
                    ghost_stencil_top2(dz, n2 - nghost_z, 0.0, bctype_top);
                }
                ghost_reduce_bot2(nghost_z - 1);
                ghost_reduce_top2(n2 - nghost_z);
                {
                    k = nghost_z;
                    tridag(&a[k], &b[k], &c[k], &r[k], &x[k], nz);
                }
                ghost_fill_bot2(nghost_z - 1);
                ghost_fill_top2(n2 - nghost_z);
                for (k = 0; k < n2; k++) {
                    int ixiyk = (local_ix * local_ny + local_iy) * n2 + k;
                    cout[ixiyk][rc] = x[k];
                }
            }
        }
}


void solve_helmholtz_u4(fftw_complex cout[],
                        int nx, int local_nx, int local_x_start,
                        int ny, int local_ny, int local_y_start,
                        int nz, int nghost_z,
                        double dx, double dy, double dz[], double nudt,
                        double avg_bot, double avg_top,
                        int bctype_bot, int bctype_top)
{
    check(order == FOURTH);

    int n2 = nz + 2 * nghost_z;

    int local_ix, local_iy, rc, k;

    for (k = 0; k < n2; k++) x[k] = 0.0;

    for (local_ix = 0; local_ix < local_nx; local_ix++)
        for (local_iy = 0; local_iy < local_ny; local_iy++) {

            int ix = local_ix + local_x_start;
            int iy = local_iy + local_y_start;
            double cosx1 = cos(2.0 * M_PI * 1.0 * (double) ix / (double) nx);
            double cosx2 = cos(2.0 * M_PI * 2.0 * (double) ix / (double) nx);
            double cosx3 = cos(2.0 * M_PI * 3.0 * (double) ix / (double) nx);
            double cosy1 = cos(2.0 * M_PI * 1.0 * (double) iy / (double) ny);
            double cosy2 = cos(2.0 * M_PI * 2.0 * (double) iy / (double) ny);
            double cosy3 = cos(2.0 * M_PI * 3.0 * (double) iy / (double) ny);
            double lamx = (2.0 * a2 * a2 + 2.0 * a1 * a1)
                        + (4.0 * a1 * a2 - 2.0 * a1 * a1) * cosx1
                        + (-4.0 * a1 * a2) * cosx2
                        + (-2.0 * a2 * a2) * cosx3;
            double lamy = (2.0 * a2 * a2 + 2.0 * a1 * a1)
                        + (4.0 * a1 * a2 - 2.0 * a1 * a1) * cosy1
                        + (-4.0 * a1 * a2) * cosy2
                        + (-2.0 * a2 * a2) * cosy3;

            for (rc = 0; rc < 2; rc++) {
                for (k = nghost_z; k < n2 - nghost_z; k++) {
                    double dzp3h = 0.5 * (dz[k + 2] + dz[k + 1]);
                    double dzp1h = 0.5 * (dz[k + 1] + dz[k]);
                    double dzm1h = 0.5 * (dz[k - 1] + dz[k]);
                    double dzm3h = 0.5 * (dz[k - 2] + dz[k - 1]);
                    double dvol = dx * dy * dz[k];
                    a[k] = + a2 * a2 * dvol * nudt / (dzm3h * dz[k]);
                    g[k] = + a2 * a2 * dvol * nudt / (dzp3h * dz[k]);
                    b[k] = + a1 * a2 * dvol * nudt / (dzm1h * dz[k])
                           + a1 * a2 * dvol * nudt / (dzm3h * dz[k]);
                    f[k] = + a1 * a2 * dvol * nudt / (dzp1h * dz[k])
                           + a1 * a2 * dvol * nudt / (dzp3h * dz[k]);
                    c[k] = - a1 * a2 * dvol * nudt / (dzp1h * dz[k])
                           - a1 * a2 * dvol * nudt / (dzm3h * dz[k])
                           + a1 * a1 * dvol * nudt / (dzm1h * dz[k]);
                    e[k] = - a1 * a2 * dvol * nudt / (dzm1h * dz[k])
                           - a1 * a2 * dvol * nudt / (dzp3h * dz[k])
                           + a1 * a1 * dvol * nudt / (dzp1h * dz[k]);
                    d[k] = - a2 * a2 * dvol * nudt / (dzm3h * dz[k])
                           - a2 * a2 * dvol * nudt / (dzp3h * dz[k])
                           - a1 * a1 * dvol * nudt / (dzm1h * dz[k])
                           - a1 * a1 * dvol * nudt / (dzp1h * dz[k])
                           - lamx    * dvol * nudt / (dx    * dx)
                           - lamy    * dvol * nudt / (dy    * dy)
                                     + dvol;
                    int ixiyk = (local_ix * local_ny + local_iy) * n2 + k;
                    r[k] = cout[ixiyk][rc] * dvol;
                }
                if (ix == 0 && iy == 0 && rc == 0) {
                    ghost_stencil_bot4(dz, nghost_z - 1,  avg_bot, bctype_bot);
                    ghost_stencil_top4(dz, n2 - nghost_z, avg_top, bctype_top);
                }
                else {
                    ghost_stencil_bot4(dz, nghost_z - 1,  0.0, bctype_bot);
                    ghost_stencil_top4(dz, n2 - nghost_z, 0.0, bctype_top);
                }
                ghost_reduce_bot4(nghost_z - 1);
                ghost_reduce_top4(n2 - nghost_z);
                {
                    k = nghost_z;
                    heptdag(&a[k], &b[k], &c[k], &d[k], &e[k],
                            &f[k], &g[k], &r[k], &x[k], nz);
                }
                ghost_fill_bot4(nghost_z - 1);
                ghost_fill_top4(n2 - nghost_z);
                for (k = 0; k < n2; k++) {
                    int ixiyk = (local_ix * local_ny + local_iy) * n2 + k;
                    cout[ixiyk][rc] = x[k];
                }
            }
        }
}


void ghost_stencil_bot2(double dz[], int k, double bc, int bctype)
{
    check(order == SECOND);
    check(bctype == NEUMANU || bctype == DIRICHU);

    double dzp1h = 0.5 * (dz[k + 1] + dz[k]);

    if (bctype == NEUMANU) {
        a[k] = 0.0;
        b[k] = - 1.0 / dzp1h;
        c[k] = + 1.0 / dzp1h;
        r[k] = bc;
    } else if (bctype == DIRICHU) {
        a[k] = 0.0;
        b[k] = 0.5;
        c[k] = 0.5;
        r[k] = bc;
    }
}


void ghost_stencil_top2(double dz[], int k, double bc, int bctype)
{
    check(order == SECOND);
    check(bctype == NEUMANU || bctype == DIRICHU);

    double dzm1h = 0.5 * (dz[k - 1] + dz[k]);

    if (bctype == NEUMANU) {
        a[k] = - 1.0 / dzm1h;
        b[k] = + 1.0 / dzm1h;
        c[k] = 0.0;
        r[k] = bc;
    } else if (bctype == DIRICHU) {
        a[k] = 0.5;
        b[k] = 0.5;
        c[k] = 0.0;
        r[k] = bc;
    }
}


void ghost_reduce_bot2(int k)
{
    check(order == SECOND);

    int kp1 = k + 1;
    double fac;
    fac = a[kp1] / b[k];
    a[kp1] -= b[k] * fac;
    b[kp1] -= c[k] * fac;
    r[kp1] -= r[k] * fac;
}


void ghost_reduce_top2(int k)
{
    check(order == SECOND);

    int km1 = k - 1;
    double fac;
    fac = c[km1] / b[k];
    c[km1] -= b[k] * fac;
    b[km1] -= a[k] * fac;
    r[km1] -= r[k] * fac;
}


void ghost_fill_bot2(int k)
{
    check(order == SECOND);

    int kp1 = k + 1;
    x[k] = (c[k] * x[kp1] - r[k]) / -b[k];
}


void ghost_fill_top2(int k)
{
    check(order == SECOND);

    int km1 = k - 1;
    x[k] = (a[k] * x[km1] - r[k]) / -b[k];
}


void ghost_stencil_bot4(double dz[], int k, double bc, int bctype)
{
    check(order == FOURTH);
    check(bctype == NEUMANU || bctype == DIRICHU);

    int km2 = k - 2;
    int km1 = k - 1;
    double dzp3h = 0.5 * (dz[k + 2] + dz[k + 1]);
    double dzp1h = 0.5 * (dz[k + 1] + dz[k]);
    double dzm1h = 0.5 * (dz[k - 1] + dz[k]);

    a[k] = 0.0;
    b[k] = - a2 / dzm1h;
    c[k] = - a1 / dzm1h - 2.0 * - a2 / dzp1h;
    d[k] = + a1 / dzm1h - 2.0 * - a1 / dzp1h - a2 / dzp3h;
    e[k] = + a2 / dzm1h - 2.0 * + a1 / dzp1h - a1 / dzp3h;
    f[k] =              - 2.0 * + a2 / dzp1h + a1 / dzp3h;
    g[k] =                                   + a2 / dzp3h;
    r[k] = 0.0;

    if (bctype == NEUMANU) {
        a[km1] = 0.0;
        b[km1] = 0.0;
        c[km1] = 0.0;
        d[km1] = - a2 / dzp1h;
        e[km1] = - a1 / dzp1h;
        f[km1] = + a1 / dzp1h;
        g[km1] = + a2 / dzp1h;
        r[km1] = bc;
    } else if (bctype == DIRICHU) {
        a[km1] = 0.0;
        b[km1] = 0.0;
        c[km1] = 0.0;
        d[km1] = 0.5 * a2;
        e[km1] = 0.5 * (a1 + 2.0 * a2);
        f[km1] = 0.5 * (a1 + 2.0 * a2);
        g[km1] = 0.5 * a2;
        r[km1] = bc;
    }

    a[km2] = 0.0;
    b[km2] = 0.0;
    c[km2] = 0.0;
    d[km2] = 0.0;
    e[km2] = 0.0;
    f[km2] = - 1.0;
    g[km2] = + 1.0;
    r[km2] = 0.0;
}


void ghost_stencil_top4(double dz[], int k, double bc, int bctype)
{
    check(order == FOURTH);
    check(bctype == NEUMANU || bctype == DIRICHU);

    int kp2 = k + 2;
    int kp1 = k + 1;
    double dzp1h = 0.5 * (dz[k + 1] + dz[k]);
    double dzm1h = 0.5 * (dz[k - 1] + dz[k]);
    double dzm3h = 0.5 * (dz[k - 2] + dz[k - 1]);

    a[k] = - a2 / dzm3h;
    b[k] = - a1 / dzm3h - 2.0 * - a2 / dzm1h;
    c[k] = + a1 / dzm3h - 2.0 * - a1 / dzm1h - a2 / dzp1h;
    d[k] = + a2 / dzm3h - 2.0 * + a1 / dzm1h - a1 / dzp1h;
    e[k] =              - 2.0 * + a2 / dzm1h + a1 / dzp1h;
    f[k] =                                   + a2 / dzp1h;
    g[k] = 0.0;
    r[k] = 0.0;

    if (bctype == NEUMANU) {
        a[kp1] = - a2 / dzm1h;
        b[kp1] = - a1 / dzm1h;
        c[kp1] = + a1 / dzm1h;
        d[kp1] = + a2 / dzm1h;
        e[kp1] = 0.0;
        f[kp1] = 0.0;
        g[kp1] = 0.0;
        r[kp1] = bc;
    } else if (bctype == DIRICHU) {
        a[kp1] = 0.5 * a2;
        b[kp1] = 0.5 * (a1 + 2.0 * a2);
        c[kp1] = 0.5 * (a1 + 2.0 * a2);
        d[kp1] = 0.5 * a2;
        e[kp1] = 0.0;
        f[kp1] = 0.0;
        g[kp1] = 0.0;
        r[kp1] = bc;
    }

    a[kp2] = - 1.0;
    b[kp2] = + 1.0;
    c[kp2] = 0.0;
    d[kp2] = 0.0;
    e[kp2] = 0.0;
    f[kp2] = 0.0;
    g[kp2] = 0.0;
    r[kp2] = 0.0;
}


void ghost_reduce_bot4(int k)
{
    check(order == FOURTH);

    // 0 0 f g
    // 0 d e f g
    // b c d e f g
    // a b c d e f g
    //   a b c d e f g
    //     a b c d e f g
    int km2 = k - 2;
    int km1 = k - 1;
    int kp1 = k + 1;
    int kp2 = k + 2;
    int kp3 = k + 3;
    double fac;
    fac = a[kp1] / b[k];
    a[kp1] -= b[k] * fac;
    b[kp1] -= c[k] * fac;
    c[kp1] -= d[k] * fac;
    d[kp1] -= e[k] * fac;
    e[kp1] -= f[k] * fac;
    f[kp1] -= g[k] * fac;
    fac = b[kp1] / d[km1];
    b[kp1] -= d[km1] * fac;
    c[kp1] -= e[km1] * fac;
    d[kp1] -= f[km1] * fac;
    e[kp1] -= g[km1] * fac;
    r[kp1] -= r[km1] * fac;
    fac = c[kp1] / f[km2];
    c[kp1] -= f[km2] * fac;
    d[kp1] -= g[km2] * fac;
    fac = a[kp2] / d[km1];
    a[kp2] -= d[km1] * fac;
    b[kp2] -= e[km1] * fac;
    c[kp2] -= f[km1] * fac;
    d[kp2] -= g[km1] * fac;
    r[kp2] -= r[km1] * fac;
    fac = b[kp2] / f[km2];
    b[kp2] -= f[km2] * fac;
    c[kp2] -= g[km2] * fac;
    fac = a[kp3] / f[km2];
    a[kp3] -= f[km2] * fac;
    b[kp3] -= g[km2] * fac;
}


void ghost_reduce_top4(int k)
{
    check(order == FOURTH);

    // a b c d e f g
    //   a b c d e f g
    //     a b c d e f g
    //       a b c d e f
    //         a b c d 0
    //           a b 0 0
    int km3 = k - 3;
    int km2 = k - 2;
    int km1 = k - 1;
    int kp1 = k + 1;
    int kp2 = k + 2;
    double fac;
    fac = g[km1] / f[k];
    g[km1] -= f[k] * fac;
    f[km1] -= e[k] * fac;
    e[km1] -= d[k] * fac;
    d[km1] -= c[k] * fac;
    c[km1] -= b[k] * fac;
    b[km1] -= a[k] * fac;
    fac = f[km1] / d[kp1];
    f[km1] -= d[kp1] * fac;
    e[km1] -= c[kp1] * fac;
    d[km1] -= b[kp1] * fac;
    c[km1] -= a[kp1] * fac;
    r[km1] -= r[kp1] * fac;
    fac = e[km1] / b[kp2];
    e[km1] -= b[kp2] * fac;
    d[km1] -= a[kp2] * fac;
    fac = g[km2] / d[kp1];
    g[km2] -= d[kp1] * fac;
    f[km2] -= c[kp1] * fac;
    e[km2] -= b[kp1] * fac;
    d[km2] -= a[kp1] * fac;
    r[km2] -= r[kp1] * fac;
    fac = f[km2] / b[kp2];
    f[km2] -= b[kp2] * fac;
    e[km2] -= a[kp2] * fac;
    fac = g[km3] / b[kp2];
    g[km3] -= b[kp2] * fac;
    f[km3] -= a[kp2] * fac;
}


void ghost_fill_bot4(int k)
{
    check(order == FOURTH);

    int km2 = k - 2;
    int km1 = k - 1;
    int kp1 = k + 1;
    int kp2 = k + 2;
    int kp3 = k + 3;
    x[k]   = (g[km2] * x[kp1]) / -f[km2];
    x[km1] = (e[km1] * x[k]
            + f[km1] * x[kp1]
            + g[km1] * x[kp2]
                     - r[km1]) / -d[km1];
    x[km2] = (c[k]   * x[km1]
            + d[k]   * x[k]
            + e[k]   * x[kp1]
            + f[k]   * x[kp2]
            + g[k]   * x[kp3]) / -b[k];
}


void ghost_fill_top4(int k)
{
    check(order == FOURTH);

    int km3 = k - 3;
    int km2 = k - 2;
    int km1 = k - 1;
    int kp1 = k + 1;
    int kp2 = k + 2;
    x[k]   = (a[kp2] * x[km1]) / -b[kp2];
    x[kp1] = (a[kp1] * x[km2]
            + b[kp1] * x[km1]
            + c[kp1] * x[k]
                     - r[kp1]) / -d[kp1];
    x[kp2] = (a[k]   * x[km3]
            + b[k]   * x[km2]
            + c[k]   * x[km1]
            + d[k]   * x[k]
            + e[k]   * x[kp1]) / -f[k];
}


void diffuseu_initialize2(int nz, int nghost_z)
{
    check(nz >= 2);
    check(nghost_z >= 1);

    int n2 = nz + 2 * nghost_z;

    a = (double *) malloc(sizeof(double) * n2);
    b = (double *) malloc(sizeof(double) * n2);
    c = (double *) malloc(sizeof(double) * n2);
    r = (double *) malloc(sizeof(double) * n2);
    x = (double *) malloc(sizeof(double) * n2);

    check(a != NULL);
    check(b != NULL);
    check(c != NULL);
    check(r != NULL);
    check(x != NULL);

    order = SECOND;
}


void diffuseu_initialize4(int nz, int nghost_z)
{
    check(nz >= 6);
    check(nghost_z >= 3);

    int n2 = nz + 2 * nghost_z;

    a = (double *) malloc(sizeof(double) * n2);
    b = (double *) malloc(sizeof(double) * n2);
    c = (double *) malloc(sizeof(double) * n2);
    d = (double *) malloc(sizeof(double) * n2);
    e = (double *) malloc(sizeof(double) * n2);
    f = (double *) malloc(sizeof(double) * n2);
    g = (double *) malloc(sizeof(double) * n2);
    r = (double *) malloc(sizeof(double) * n2);
    x = (double *) malloc(sizeof(double) * n2);

    check(a != NULL);
    check(b != NULL);
    check(c != NULL);
    check(d != NULL);
    check(e != NULL);
    check(f != NULL);
    check(g != NULL);
    check(r != NULL);
    check(x != NULL);

    order = FOURTH;
}


void diffuseu_finalize2(void)
{
    check(order == SECOND);

    free(a);
    free(b);
    free(c);
    free(r);
    free(x);
}


void diffuseu_finalize4(void)
{
    check(order == FOURTH);

    free(a);
    free(b);
    free(c);
    free(d);
    free(e);
    free(f);
    free(g);
    free(r);
    free(x);
}
