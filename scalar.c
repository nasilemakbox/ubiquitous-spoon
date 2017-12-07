#include "check.h"
#include "fourth.h"
#include "scalar.h"


void add_buoyancy2(double c[],
                   double fu[], double fv[], double fw[],
                   double betg_x, double betg_y, double betg_z,
                   int local_nx, int local_ny, int nz,
                   int nghost_x, int nghost_y, int nghost_z)
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0 - 1; i++)
        for (j = 0; j < n1 - 1; j++)
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                fu[ijk] += 0.5 * (c[ijk] + c[ip1]) * betg_x;
                fv[ijk] += 0.5 * (c[ijk] + c[jp1]) * betg_y;
            }

    for (i = 0; i < n0 - 1; i++)
        for (j = 0; j < n1 - 1; j++)
            for (k = nghost_z; k < n2 - nghost_z - 1; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                fw[ijk] += 0.5 * (c[ijk] + c[kp1]) * betg_z;
            }
}


void add_buoyancy4(double c[],
                   double fu[], double fv[], double fw[],
                   double betg_x, double betg_y, double betg_z,
                   int local_nx, int local_ny, int nz,
                   int nghost_x, int nghost_y, int nghost_z)
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 1; i < n0 - 2; i++)
        for (j = 1; j < n1 - 2; j++)
            for (k = nghost_z; k < n2 - nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                int ip2 = ((i + 2) * n1 + j) * n2 + k;
                int jp2 = (i * n1 + (j + 2)) * n2 + k;
                double cphx = b1 * (c[ijk] + c[ip1]) + b2 * (c[im1] + c[ip2]);
                double cphy = b1 * (c[ijk] + c[jp1]) + b2 * (c[jm1] + c[jp2]);
                fu[ijk] += cphx * betg_x;
                fv[ijk] += cphy * betg_y;
            }

    for (i = 1; i < n0 - 2; i++)
        for (j = 1; j < n1 - 2; j++)
            for (k = nghost_z; k < n2 - nghost_z - 1; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int kp2 = (i * n1 + j) * n2 + (k + 2);
                double cphz = b1 * (c[ijk] + c[kp1]) + b2 * (c[km1] + c[kp2]);
                fw[ijk] += cphz * betg_z;
            }
}


void advect_scalar(double mucphx[], double mvcphy[], double mwcphz[],
                   double fc[],
                   int local_nx, int local_ny, int nz,
                   int nghost_x, int nghost_y, int nghost_z,
                   double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int local_size = n0 * n1 * n2;

    int i, j, k;

    for (i = 0; i < local_size; i++) fc[i] = 0.0;

    for (i = 1; i < n0; i++)
        for (j = 1; j < n1; j++)
            for (k = nghost_z; k < n2 - nghost_z; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);

                fc[ijk] = - (
                    (mucphx[ijk] - mucphx[im1])
                  + (mvcphy[ijk] - mvcphy[jm1])
                  + (mwcphz[ijk] - mwcphz[km1])
                            ) / (dx * dy * dz[k]);
            }
}


void flux_quick(double u[], double v[], double w[],
                double c[],
                double mucphx[], double mvcphy[], double mwcphz[],
                int local_nx, int local_ny, int nz,
                int nghost_x, int nghost_y, int nghost_z,
                double dx, double dy, double dz[])
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    const double q = -0.125;

    int i, j, k;

    for (i = 1; i < n0 - 2; i++)
        for (j = 1; j < n1 - 2; j++)
            for (k = 1; k < n2 - 2; k++) {

                int ijk = (i * n1 + j) * n2 + k;
                int im1 = ((i - 1) * n1 + j) * n2 + k;
                int jm1 = (i * n1 + (j - 1)) * n2 + k;
                int km1 = (i * n1 + j) * n2 + (k - 1);
                int ip1 = ((i + 1) * n1 + j) * n2 + k;
                int jp1 = (i * n1 + (j + 1)) * n2 + k;
                int kp1 = (i * n1 + j) * n2 + (k + 1);
                int ip2 = ((i + 2) * n1 + j) * n2 + k;
                int jp2 = (i * n1 + (j + 2)) * n2 + k;
                int kp2 = (i * n1 + j) * n2 + (k + 2);

                double mu = dy * dz[k] * u[ijk];
                double mv = dx * dz[k] * v[ijk];
                double mw = dx * dy    * w[ijk];

                double cphx = 0.5 * (c[ijk] + c[ip1]);
                double cphy = 0.5 * (c[ijk] + c[jp1]);
                double cphz = 0.5 * (c[ijk] + c[kp1]);

                cphx += mu > 0.0 ? q * (c[ip1] - 2.0 * c[ijk]  + c[im1])
                                 : q * (c[ip2] - 2.0 * c[ip1]  + c[ijk]);
                cphy += mv > 0.0 ? q * (c[jp1] - 2.0 * c[ijk]  + c[jm1])
                                 : q * (c[jp2] - 2.0 * c[jp1]  + c[ijk]);
                cphz += mw > 0.0 ? q * (c[kp1] - 2.0 * c[ijk]  + c[km1])
                                 : q * (c[kp2] - 2.0 * c[kp1]  + c[ijk]);

                mucphx[ijk] = mu * cphx;
                mvcphy[ijk] = mv * cphy;
                mwcphz[ijk] = mw * cphz;
            }
}
