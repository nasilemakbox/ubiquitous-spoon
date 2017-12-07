#include "check.h"
#include "cases.h"
#include "channel.h"
#include "chanfast.h"


int main(int argc, char **argv)
{
    int p_row = 2;
    int p_col = 2;
    int nghost_x = 3;
    int nghost_y = 3;
    int nghost_z = 3;
    int nx = 128;
    int ny = 128;
    int nz = 128;

    MPI_Init(&argc, &argv);

    int nrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);

    if (nrank == 0)
        printf("p_row=%d p_col=%d nx=%d ny=%d nz=%d nghost_x=%d nghost_y=%d nghost_z=%d\n",
            p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z);
    MPI_Barrier(MPI_COMM_WORLD);

    //case0(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Pressure accuracy 2
    //case1(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Pressure Poisson 2
    //case2(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // u-diffuse accuracy 2
    //case3(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Time step u-diffuse 2
    //case4(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // w-diffuse accuracy 2
    //case5(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Time step w-diffuse 2
    //case6(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Advection 2
    //case7(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Pressure accuracy 4
    //case8(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Pressure Poisson 4
    //case9(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // u-diffuse accuracy 4
    //case10(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Time step u-diffuse 4
    //case11(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // w-diffuse accuracy 4
    //case12(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Time step w-diffuse 4
    //case13(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Advection 4
    //case14(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Read/write
    //case15(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Moments
    //case16(p_row, p_col, nx, ny, nz, nghost_x, nghost_y, nghost_z); // Spectra
    {
        double lx = 4.0 * M_PI, ly = 4.0 / 3.0 * M_PI, lz = 2.0;
        double dpdx = -1.0, nu = 1.0 / 180.0, dt = 0.002, want_cfl = 1.0;
        double want_uvolavg = 15.63, want_cvolavg = 0.0;
        int cflmode = 1, uvolavgmode = 1, cvolavgmode = 1;
        double setuframe = want_uvolavg, setvframe = 0.0;
        double betg_x = 0.0, betg_y = 0.0, betg_z = 0.0, prandtl = 1.0;
        int it = 17000, nt = 18000, istat = nt;
        //channel2(
        //channel4(
        //chanfast2(
        chanfast4(
            p_row, p_col, nx, ny, nz, it, nt, istat, nghost_x, nghost_y, nghost_z,
            lx, ly, lz, dt, want_cfl, cflmode, nu, prandtl, dpdx,
            want_uvolavg, uvolavgmode, want_cvolavg, cvolavgmode,
            setuframe, setvframe, betg_x, betg_y, betg_z);
    }
    MPI_Finalize();

    return EXIT_SUCCESS;
}
