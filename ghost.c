#include "check.h"
#include "ghost.h"


void ghost_odd_z(double u[],
                 int local_nx, int local_ny, int nz,
                 int nghost_x, int nghost_y, int nghost_z)
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++) {
            for (k = 0; k < nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int ijk1 = (i * n1 + j) * n2 + 2 * nghost_z - k - 1;
                u[ijk] = -u[ijk1];
            }
            for (k = n2 - nghost_z; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int ijk1 = (i * n1 + j) * n2 + 2 * (n2 - nghost_z) - k - 1;
                u[ijk] = -u[ijk1];
            }
        }
}


void ghost_even_z(double u[],
                  int local_nx, int local_ny, int nz,
                  int nghost_x, int nghost_y, int nghost_z)
{
    int n0 = local_nx + 2 * nghost_x;
    int n1 = local_ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;

    int i, j, k;

    for (i = 0; i < n0; i++)
        for (j = 0; j < n1; j++) {
            for (k = 0; k < nghost_z; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int ijk1 = (i * n1 + j) * n2 + 2 * nghost_z - k - 1;
                u[ijk] = u[ijk1];
            }
            for (k = n2 - nghost_z; k < n2; k++) {
                int ijk = (i * n1 + j) * n2 + k;
                int ijk1 = (i * n1 + j) * n2 + 2 * (n2 - nghost_z) - k - 1;
                u[ijk] = u[ijk1];
            }
        }
}
