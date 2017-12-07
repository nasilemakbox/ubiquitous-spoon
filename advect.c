#include "check.h"
#include "fourth.h"
#include "advect.h"


void massfluxx2(double u[], double v[], double w[],
                double muphx[], double mvphx[], double mwphx[],
                int local_nx, int local_ny, int nz,
                int nghost_x, int nghost_y, int nghost_z,
                double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0 - 1; i++)
        for (j = 0; j < n1 - 1; j++)
            for (k = 0; k < n2 - 1; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int ip1 = ((i + 1) * n1 + j) * n2 + k;

                double dzm0 = dz[k];
                muphx[ijk] = 0.5 * (dy * dzm0 * u[ijk] + dy * dzm0 * u[ip1]);
                mvphx[ijk] = 0.5 * (dx * dzm0 * v[ijk] + dx * dzm0 * v[ip1]);
                mwphx[ijk] = 0.5 * (dx * dy   * w[ijk] + dx * dy   * w[ip1]);
            }
}


void massfluxy2(double u[], double v[], double w[],
                double muphy[], double mvphy[], double mwphy[],
                int local_nx, int local_ny, int nz,
                int nghost_x, int nghost_y, int nghost_z,
                double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0 - 1; i++)
        for (j = 0; j < n1 - 1; j++)
            for (k = 0; k < n2 - 1; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;

                double dzm0 = dz[k];
                muphy[ijk] = 0.5 * (dy * dzm0 * u[ijk] + dy * dzm0 * u[jp1]);
                mvphy[ijk] = 0.5 * (dx * dzm0 * v[ijk] + dx * dzm0 * v[jp1]);
                mwphy[ijk] = 0.5 * (dx * dy   * w[ijk] + dx * dy   * w[jp1]);
            }
}


void massfluxz2(double u[], double v[], double w[],
                double muphz[], double mvphz[], double mwphz[],
                int local_nx, int local_ny, int nz,
                int nghost_x, int nghost_y, int nghost_z,
                double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0 - 1; i++)
        for (j = 0; j < n1 - 1; j++)
            for (k = 0; k < n2 - 1; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);

                double dzm0 = dz[k];
                double dzp1 = dz[k + 1];
                muphz[ijk] = 0.5 * (dy * dzm0 * u[ijk] + dy * dzp1 * u[kp1]);
                mvphz[ijk] = 0.5 * (dx * dzm0 * v[ijk] + dx * dzp1 * v[kp1]);
                mwphz[ijk] = 0.5 * (dx * dy   * w[ijk] + dx * dy   * w[kp1]);
            }
}


void massfluxx4(double u[], double v[], double w[],
                double muphx[], double mvphx[], double mwphx[],
                int local_nx, int local_ny, int nz,
                int nghost_x, int nghost_y, int nghost_z,
                double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++) {
            {
                k = nghost_z - 1;
                int km2 = (i * n1 + j) * n2 + (k - 2);
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                u[ijk] = u[kp1];
                v[ijk] = v[kp1];
                w[km1] = 2.0 * w[ijk] - w[kp1];
                w[km2] = 2.0 * w[ijk] - w[kp2];
            }
            {
                k = n2 - nghost_z;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = u[km1];
                v[ijk] = v[km1];
            }
            {
                k = n2 - nghost_z - 1;
                int km2 = (i * n1 + j) * n2 + (k - 2);
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                w[kp1] = 2.0 * w[ijk] - w[km1];
                w[kp2] = 2.0 * w[ijk] - w[km2];
            }
        }

    for (i = 1; i < n0 - 2; i++)
        for (j = 1; j < n1 - 2; j++)
            for (k = 1; k < n2 - 2; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int ip2 = ((i + 2) * n1 + j) * n2 + k;

                double dzm0 = dz[k];
                muphx[ijk] = b1 * (dy * dzm0 * u[ijk] + dy * dzm0 * u[ip1])
                           + b2 * (dy * dzm0 * u[im1] + dy * dzm0 * u[ip2]);
                mvphx[ijk] = b1 * (dx * dzm0 * v[ijk] + dx * dzm0 * v[ip1])
                           + b2 * (dx * dzm0 * v[im1] + dx * dzm0 * v[ip2]);
                mwphx[ijk] = b1 * (dx * dy   * w[ijk] + dx * dy   * w[ip1])
                           + b2 * (dx * dy   * w[im1] + dx * dy   * w[ip2]);
            }
}


void massfluxy4(double u[], double v[], double w[],
                double muphy[], double mvphy[], double mwphy[],
                int local_nx, int local_ny, int nz,
                int nghost_x, int nghost_y, int nghost_z,
                double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++) {
            {
                k = nghost_z - 1;
                int km2 = (i * n1 + j) * n2 + (k - 2);
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                u[ijk] = u[kp1];
                v[ijk] = v[kp1];
                w[km1] = 2.0 * w[ijk] - w[kp1];
                w[km2] = 2.0 * w[ijk] - w[kp2];
            }
            {
                k = n2 - nghost_z;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = u[km1];
                v[ijk] = v[km1];
            }
            {
                k = n2 - nghost_z - 1;
                int km2 = (i * n1 + j) * n2 + (k - 2);
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                w[kp1] = 2.0 * w[ijk] - w[km1];
                w[kp2] = 2.0 * w[ijk] - w[km2];
            }
        }

    for (i = 1; i < n0 - 2; i++)
        for (j = 1; j < n1 - 2; j++)
            for (k = 1; k < n2 - 2; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                int jp2 = (i * n1 + (j + 2)) * n2 + k;

                double dzm0 = dz[k];
                muphy[ijk] = b1 * (dy * dzm0 * u[ijk] + dy * dzm0 * u[jp1])
                           + b2 * (dy * dzm0 * u[jm1] + dy * dzm0 * u[jp2]);
                mvphy[ijk] = b1 * (dx * dzm0 * v[ijk] + dx * dzm0 * v[jp1])
                           + b2 * (dx * dzm0 * v[jm1] + dx * dzm0 * v[jp2]);
                mwphy[ijk] = b1 * (dx * dy   * w[ijk] + dx * dy   * w[jp1])
                           + b2 * (dx * dy   * w[jm1] + dx * dy   * w[jp2]);
            }
}


void massfluxz4(double u[], double v[], double w[],
                double muphz[], double mvphz[], double mwphz[],
                int local_nx, int local_ny, int nz,
                int nghost_x, int nghost_y, int nghost_z,
                double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++) {
            {
                k = nghost_z - 1;
                int km2 = (i * n1 + j) * n2 + (k - 2);
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                u[ijk] = u[kp1];
                v[ijk] = v[kp1];
                w[km1] = 2.0 * w[ijk] - w[kp1];
                w[km2] = 2.0 * w[ijk] - w[kp2];
            }
            {
                k = n2 - nghost_z;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                u[ijk] = u[km1];
                v[ijk] = v[km1];
            }
            {
                k = n2 - nghost_z - 1;
                int km2 = (i * n1 + j) * n2 + (k - 2);
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                w[kp1] = 2.0 * w[ijk] - w[km1];
                w[kp2] = 2.0 * w[ijk] - w[km2];
            }
        }

    for (i = 1; i < n0 - 2; i++)
        for (j = 1; j < n1 - 2; j++)
            for (k = 1; k < n2 - 2; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int kp2 = (i * n1 + j) * n2 + (k + 2);

                double dzm1 = dz[k - 1];
                double dzm0 = dz[k];
                double dzp1 = dz[k + 1];
                double dzp2 = dz[k + 2];
                muphz[ijk] = b1 * (dy * dzm0 * u[ijk] + dy * dzp1 * u[kp1])
                           + b2 * (dy * dzm1 * u[km1] + dy * dzp2 * u[kp2]);
                mvphz[ijk] = b1 * (dx * dzm0 * v[ijk] + dx * dzp1 * v[kp1])
                           + b2 * (dx * dzm1 * v[km1] + dx * dzp2 * v[kp2]);
                mwphz[ijk] = b1 * (dx * dy   * w[ijk] + dx * dy   * w[kp1])
                           + b2 * (dx * dy   * w[km1] + dx * dy   * w[kp2]);
            }
}


void advectu2(double u[],
              double muphx[], double mvphx[], double mwphx[],
              double fu[],
              int local_nx, int local_ny, int nz,
              int nghost_x, int nghost_y, int nghost_z,
              double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int local_size = n0 * n1 * n2;

    int i, j, k;

    for (i = 0; i < local_size; i++) fu[i] = 0.0;

    double ubc_bot, ubc_top;

    for (i = 1; i < n0 - 1; i++)
        for (j = 1; j < n1 - 1; j++) {
            {
                k = nghost_z - 1;
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                ubc_bot = 0.5 * (u[ijk] + u[kp1]);
            }
            {
                k = n2 - nghost_z;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                ubc_top = 0.5 * (u[km1] + u[ijk]);
            }
            for (k = nghost_z; k < n2 - nghost_z; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);

                double muuphx = muphx[ijk] * 0.5 * (u[ijk] + u[ip1]);
                double mvuphy = mvphx[ijk] * 0.5 * (u[ijk] + u[jp1]);
                double mwuphz = mwphx[ijk] * 0.5 * (u[ijk] + u[kp1]);
                double muumhx = muphx[im1] * 0.5 * (u[ijk] + u[im1]);
                double mvumhy = mvphx[jm1] * 0.5 * (u[ijk] + u[jm1]);
                double mwumhz = mwphx[km1] * 0.5 * (u[ijk] + u[km1]);

                if (k == n2 - nghost_z - 1) {
                    mwuphz = mwphx[ijk] * ubc_top;
                }
                if (k == nghost_z) {
                    mwumhz = mwphx[km1] * ubc_bot;
                }

                fu[ijk] = - (
                    (muuphx - muumhx)
                  + (mvuphy - mvumhy)
                  + (mwuphz - mwumhz)
                            ) / (dx * dy * dz[k]);
            }
        }
}


void advectv2(double v[],
              double muphy[], double mvphy[], double mwphy[],
              double fv[],
              int local_nx, int local_ny, int nz,
              int nghost_x, int nghost_y, int nghost_z,
              double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int local_size = n0 * n1 * n2;

    int i, j, k;

    for (i = 0; i < local_size; i++) fv[i] = 0.0;

    double vbc_bot, vbc_top;

    for (i = 1; i < n0 - 1; i++)
        for (j = 1; j < n1 - 1; j++) {
            {
                k = nghost_z - 1;
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                vbc_bot = 0.5 * (v[ijk] + v[kp1]);
            }
            {
                k = n2 - nghost_z;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                vbc_top = 0.5 * (v[km1] + v[ijk]);
            }
            for (k = nghost_z; k < n2 - nghost_z; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);

                double muvphx = muphy[ijk] * 0.5 * (v[ijk] + v[ip1]);
                double mvvphy = mvphy[ijk] * 0.5 * (v[ijk] + v[jp1]);
                double mwvphz = mwphy[ijk] * 0.5 * (v[ijk] + v[kp1]);
                double muvmhx = muphy[im1] * 0.5 * (v[ijk] + v[im1]);
                double mvvmhy = mvphy[jm1] * 0.5 * (v[ijk] + v[jm1]);
                double mwvmhz = mwphy[km1] * 0.5 * (v[ijk] + v[km1]);

                if (k == n2 - nghost_z - 1) {
                    mwvphz = mwphy[ijk] * vbc_top;
                }
                if (k == nghost_z) {
                    mwvmhz = mwphy[km1] * vbc_bot;
                }

                fv[ijk] = - (
                    (muvphx - muvmhx)
                  + (mvvphy - mvvmhy)
                  + (mwvphz - mwvmhz)
                            ) / (dx * dy * dz[k]);
            }
        }
}


void advectw2(double w[],
              double muphz[], double mvphz[], double mwphz[],
              double fw[],
              int local_nx, int local_ny, int nz,
              int nghost_x, int nghost_y, int nghost_z,
              double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int local_size = n0 * n1 * n2;

    int i, j, k;

    for (i = 0; i < local_size; i++) fw[i] = 0.0;

    for (i = 1; i < n0 - 1; i++)
        for (j = 1; j < n1 - 1; j++)
            for (k = nghost_z; k < n2 - nghost_z - 1; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                double dzp1h = 0.5 * (dz[k + 1] + dz[k]);

                double muwphx = muphz[ijk] * 0.5 * (w[ijk] + w[ip1]);
                double mvwphy = mvphz[ijk] * 0.5 * (w[ijk] + w[jp1]);
                double mwwphz = mwphz[ijk] * 0.5 * (w[ijk] + w[kp1]);
                double muwmhx = muphz[im1] * 0.5 * (w[ijk] + w[im1]);
                double mvwmhy = mvphz[jm1] * 0.5 * (w[ijk] + w[jm1]);
                double mwwmhz = mwphz[km1] * 0.5 * (w[ijk] + w[km1]);

                fw[ijk] = - (
                    (muwphx - muwmhx)
                  + (mvwphy - mvwmhy)
                  + (mwwphz - mwwmhz)
                            ) / (dx * dy * dzp1h);
            }
}


void advectu4(double u[],
              double muphx[], double mvphx[], double mwphx[],
              double fu[],
              int local_nx, int local_ny, int nz,
              int nghost_x, int nghost_y, int nghost_z,
              double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int local_size = n0 * n1 * n2;

    int i, j, k;

    for (i = 0; i < local_size; i++) fu[i] = 0.0;

    double ubc_bot, ubc_top;

    for (i = 3; i < n0 - 3; i++)
        for (j = 3; j < n1 - 3; j++) {
            {
                k = nghost_z - 1;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                ubc_bot =          0.5 * a2 * (u[km1] + u[kp2])
                    + 0.5 * (a1 + 2.0 * a2) * (u[ijk] + u[kp1]);
            }
            {
                k = n2 - nghost_z;
                int km2 = (i * n1 + j) * n2 + (k - 2);
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                ubc_top =          0.5 * a2 * (u[km2] + u[kp1])
                    + 0.5 * (a1 + 2.0 * a2) * (u[km1] + u[ijk]);
            }
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
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                int im3 = ((i - 3) * n1 + j) * n2 + k;
                int jm3 = (i * n1 + (j - 3)) * n2 + k;
                int km3 = (i * n1 + j) * n2 + (k - 3);
                int ip3 = ((i + 3) * n1 + j) * n2 + k;
                int jp3 = (i * n1 + (j + 3)) * n2 + k;
                int kp3 = (i * n1 + j) * n2 + (k + 3);

                double muuphx = muphx[ijk] * 0.5 * (u[ijk] + u[ip1]);
                double mvuphy = mvphx[ijk] * 0.5 * (u[ijk] + u[jp1]);
                double mwuphz = mwphx[ijk] * 0.5 * (u[ijk] + u[kp1]);
                double muumhx = muphx[im1] * 0.5 * (u[ijk] + u[im1]);
                double mvumhy = mvphx[jm1] * 0.5 * (u[ijk] + u[jm1]);
                double mwumhz = mwphx[km1] * 0.5 * (u[ijk] + u[km1]);

                double muup3hx = muphx[ip1] * 0.5 * (u[ijk] + u[ip3]);
                double mvup3hy = mvphx[jp1] * 0.5 * (u[ijk] + u[jp3]);
                double mwup3hz = mwphx[kp1] * 0.5 * (u[ijk] + u[kp3]);
                double muum3hx = muphx[im2] * 0.5 * (u[ijk] + u[im3]);
                double mvum3hy = mvphx[jm2] * 0.5 * (u[ijk] + u[jm3]);
                double mwum3hz = mwphx[km2] * 0.5 * (u[ijk] + u[km3]);

                if (k == n2 - nghost_z - 1) {
                    mwuphz  =   mwphx[ijk] * ubc_top;
                    mwup3hz = - mwphx[km1] * 0.5 * (u[km2] + u[kp1]) 
                        + 2.0 * mwphx[ijk] * ubc_top;
                }
                if (k == n2 - nghost_z - 2) {
                    mwup3hz =   mwphx[kp1] * ubc_top;
                }
                if (k == nghost_z) {
                    mwumhz  =   mwphx[km1] * ubc_bot;
                    mwum3hz = - mwphx[ijk] * 0.5 * (u[kp2] + u[km1])
                        + 2.0 * mwphx[km1] * ubc_bot;
                }
                if (k == nghost_z + 1) {
                    mwum3hz =   mwphx[km2] * ubc_bot;
                }

                fu[ijk] = - (
                    (a1 * (muuphx  - muumhx )
                   + a2 * (muup3hx - muum3hx))
                  + (a1 * (mvuphy  - mvumhy )
                   + a2 * (mvup3hy - mvum3hy))
                  + (a1 * (mwuphz  - mwumhz )
                   + a2 * (mwup3hz - mwum3hz))
                            ) / (dx * dy * dz[k]);
            }
        }
}


void advectv4(double v[],
              double muphy[], double mvphy[], double mwphy[],
              double fv[],
              int local_nx, int local_ny, int nz,
              int nghost_x, int nghost_y, int nghost_z,
              double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int local_size = n0 * n1 * n2;

    int i, j, k;

    for (i = 0; i < local_size; i++) fv[i] = 0.0;

    double vbc_bot, vbc_top;

    for (i = 3; i < n0 - 3; i++)
        for (j = 3; j < n1 - 3; j++) {
            {
                k = nghost_z - 1;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                vbc_bot =          0.5 * a2 * (v[km1] + v[kp2])
                    + 0.5 * (a1 + 2.0 * a2) * (v[ijk] + v[kp1]);
            }
            {
                k = n2 - nghost_z;
                int km2 = (i * n1 + j) * n2 + (k - 2);
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                vbc_top =          0.5 * a2 * (v[km2] + v[kp1])
                    + 0.5 * (a1 + 2.0 * a2) * (v[km1] + v[ijk]);
            }
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
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                int im3 = ((i - 3) * n1 + j) * n2 + k;
                int jm3 = (i * n1 + (j - 3)) * n2 + k;
                int km3 = (i * n1 + j) * n2 + (k - 3);
                int ip3 = ((i + 3) * n1 + j) * n2 + k;
                int jp3 = (i * n1 + (j + 3)) * n2 + k;
                int kp3 = (i * n1 + j) * n2 + (k + 3);

                double muvphx = muphy[ijk] * 0.5 * (v[ijk] + v[ip1]);
                double mvvphy = mvphy[ijk] * 0.5 * (v[ijk] + v[jp1]);
                double mwvphz = mwphy[ijk] * 0.5 * (v[ijk] + v[kp1]);
                double muvmhx = muphy[im1] * 0.5 * (v[ijk] + v[im1]);
                double mvvmhy = mvphy[jm1] * 0.5 * (v[ijk] + v[jm1]);
                double mwvmhz = mwphy[km1] * 0.5 * (v[ijk] + v[km1]);

                double muvp3hx = muphy[ip1] * 0.5 * (v[ijk] + v[ip3]);
                double mvvp3hy = mvphy[jp1] * 0.5 * (v[ijk] + v[jp3]);
                double mwvp3hz = mwphy[kp1] * 0.5 * (v[ijk] + v[kp3]);
                double muvm3hx = muphy[im2] * 0.5 * (v[ijk] + v[im3]);
                double mvvm3hy = mvphy[jm2] * 0.5 * (v[ijk] + v[jm3]);
                double mwvm3hz = mwphy[km2] * 0.5 * (v[ijk] + v[km3]);

                if (k == n2 - nghost_z - 1) {
                    mwvphz  =   mwphy[ijk] * vbc_top;
                    mwvp3hz = - mwphy[km1] * 0.5 * (v[km2] + v[kp1])
                        + 2.0 * mwphy[ijk] * vbc_top;
                }
                if (k == n2 - nghost_z - 2) {
                    mwvp3hz =   mwphy[kp1] * vbc_top;
                }
                if (k == nghost_z) {
                    mwvmhz  =   mwphy[km1] * vbc_bot;
                    mwvm3hz = - mwphy[ijk] * 0.5 * (v[kp2] + v[km1])
                        + 2.0 * mwphy[km1] * vbc_bot;
                }
                if (k == nghost_z + 1) {
                    mwvm3hz =   mwphy[km2] * vbc_bot;
                }

                fv[ijk] = - (
                    (a1 * (muvphx  - muvmhx )
                   + a2 * (muvp3hx - muvm3hx))
                  + (a1 * (mvvphy  - mvvmhy )
                   + a2 * (mvvp3hy - mvvm3hy))
                  + (a1 * (mwvphz  - mwvmhz )
                   + a2 * (mwvp3hz - mwvm3hz))
                            ) / (dx * dy * dz[k]);
            }
        }
}


void advectw4(double w[],
              double muphz[], double mvphz[], double mwphz[],
              double fw[],
              int local_nx, int local_ny, int nz,
              int nghost_x, int nghost_y, int nghost_z,
              double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int local_size = n0 * n1 * n2;

    int i, j, k;

    for (i = 0; i < local_size; i++) fw[i] = 0.0;

    for (i = 3; i < n0 - 3; i++)
        for (j = 3; j < n1 - 3; j++)
            for (k = nghost_z; k < n2 - nghost_z - 1; k++) {

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
                int im3 = ((i - 3) * n1 + j) * n2 + k;
                int jm3 = (i * n1 + (j - 3)) * n2 + k;
                int km3 = (i * n1 + j) * n2 + (k - 3);
                int ip3 = ((i + 3) * n1 + j) * n2 + k;
                int jp3 = (i * n1 + (j + 3)) * n2 + k;
                int kp3 = (i * n1 + j) * n2 + (k + 3);
                double dzp1h = 0.5 * (dz[k + 1] + dz[k]);

                double muwphx = muphz[ijk] * 0.5 * (w[ijk] + w[ip1]);
                double mvwphy = mvphz[ijk] * 0.5 * (w[ijk] + w[jp1]);
                double mwwphz = mwphz[ijk] * 0.5 * (w[ijk] + w[kp1]);
                double muwmhx = muphz[im1] * 0.5 * (w[ijk] + w[im1]);
                double mvwmhy = mvphz[jm1] * 0.5 * (w[ijk] + w[jm1]);
                double mwwmhz = mwphz[km1] * 0.5 * (w[ijk] + w[km1]);

                double muwp3hx = muphz[ip1] * 0.5 * (w[ijk] + w[ip3]);
                double mvwp3hy = mvphz[jp1] * 0.5 * (w[ijk] + w[jp3]);
                double mwwp3hz = mwphz[kp1] * 0.5 * (w[ijk] + w[kp3]);
                double muwm3hx = muphz[im2] * 0.5 * (w[ijk] + w[im3]);
                double mvwm3hy = mvphz[jm2] * 0.5 * (w[ijk] + w[jm3]);
                double mwwm3hz = mwphz[km2] * 0.5 * (w[ijk] + w[km3]);

                fw[ijk] = - (
                    (a1 * (muwphx  - muwmhx )
                   + a2 * (muwp3hx - muwm3hx))
                  + (a1 * (mvwphy  - mvwmhy )
                   + a2 * (mvwp3hy - mvwm3hy))
                  + (a1 * (mwwphz  - mwwmhz )
                   + a2 * (mwwp3hz - mwwm3hz))
                            ) / (dx * dy * dzp1h);
            }
}
