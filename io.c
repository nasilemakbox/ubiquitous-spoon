#include "check.h"
#include "io.h"


void io_append_ascii(double u[], int n, char filename[])
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        FILE *fp = fopen(filename, "a");
        check(fp != NULL);
        fprintf(fp, "%+e", u[0]);
        int i;
        for (i = 1; i < n; i++) fprintf(fp, " %+e", u[i]);
        fprintf(fp, "\n");
        fclose(fp);
    }
}


void io_write_ascii(double u[], int n, char filename[])
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        FILE *fp = fopen(filename, "w");
        check(fp != NULL);
        int i;
        for (i = 0; i < n; i++) fprintf(fp, "%+e\n", u[i]);
        fclose(fp);
    }
}


/*
void io_write_kslice(double u[], char filename[],
                     int local_nx, int ny, int nz,
                     int nghost_x, int nghost_y, int nghost_z,
                     int local_x_start, int kslice)
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *recvcnts, *displs;
    if (rank == 0) {
        recvcnts = (int *) malloc(sizeof(int) * size);
        displs = (int *) malloc(sizeof(int) * size);
        check(recvcnts != NULL);
        check(displs != NULL);
    }
    else {
        recvcnts = NULL;
        displs = NULL;
    }
    int local_recvcnt = local_nx * ny;
    int local_displ = local_x_start * ny;

    MPI_Gather(&local_recvcnt, 1, MPI_INT,
               recvcnts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_displ, 1, MPI_INT,
               displs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_recvcnt = 0;
    double *recvbuf;
    if (rank == 0) {
        int p;
        for (p = 0; p < size; p++) total_recvcnt += recvcnts[p];
        recvbuf = (double *) malloc(sizeof(double) * total_recvcnt);
        check(recvbuf != NULL);
    }
    else {
        recvbuf = NULL;
    }

    double *sendbuf = (double *) malloc(sizeof(double) * local_recvcnt);
    check(sendbuf != NULL);

    int n1 = ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;
    int ix, iy;

    for (ix = 0; ix < local_nx; ix++)
        for (iy = 0; iy < ny; iy++) {
            int i = ix + nghost_x;
            int j = iy + nghost_y;
            int ijk = (i * n1 + j) * n2 + kslice;
            int ixiy = ix * ny + iy;
            sendbuf[ixiy] = u[ijk];
        }

    MPI_Gatherv(sendbuf, local_recvcnt, MPI_DOUBLE,
                recvbuf, recvcnts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        io_write_ascii(recvbuf, total_recvcnt, filename);

        free(recvbuf);
        free(recvcnts);
        free(displs);
    }

    free(sendbuf);
}


void io_write_jslice(double u[], char filename[],
                     int local_nx, int ny, int nz,
                     int nghost_x, int nghost_y, int nghost_z,
                     int local_x_start, int jslice)
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *recvcnts, *displs;
    if (rank == 0) {
        recvcnts = (int *) malloc(sizeof(int) * size);
        displs = (int *) malloc(sizeof(int) * size);
        check(recvcnts != NULL);
        check(displs != NULL);
    }
    else {
        recvcnts = NULL;
        displs = NULL;
    }
    int local_recvcnt = local_nx * nz;
    int local_displ = local_x_start * nz;

    MPI_Gather(&local_recvcnt, 1, MPI_INT,
               recvcnts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_displ, 1, MPI_INT,
               displs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_recvcnt = 0;
    double *recvbuf;
    if (rank == 0) {
        int p;
        for (p = 0; p < size; p++) total_recvcnt += recvcnts[p];
        recvbuf = (double *) malloc(sizeof(double) * total_recvcnt);
        check(recvbuf != NULL);
    }
    else {
        recvbuf = NULL;
    }

    double *sendbuf = (double *) malloc(sizeof(double) * local_recvcnt);
    check(sendbuf != NULL);

    int n1 = ny + 2 * nghost_y;
    int n2 = nz + 2 * nghost_z;
    int ix, iz;

    for (ix = 0; ix < local_nx; ix++)
        for (iz = 0; iz < nz; iz++) {
            int i = ix + nghost_x;
            int k = iz + nghost_z;
            int ijk = (i * n1 + jslice) * n2 + k;
            int ixiz = ix * nz + iz;
            sendbuf[ixiz] = u[ijk];
        }

    MPI_Gatherv(sendbuf, local_recvcnt, MPI_DOUBLE,
                recvbuf, recvcnts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        io_write_ascii(recvbuf, total_recvcnt, filename);

        free(recvbuf);
        free(recvcnts);
        free(displs);
    }

    free(sendbuf);
}


void io_write_islice(double u[], char filename[],
                     int local_nx, int ny, int nz,
                     int nghost_y, int nghost_z,
                     int local_x_start, int islice)
{
    if (local_x_start <= islice && islice < local_x_start + local_nx) {

        double *buf = (double *) malloc(sizeof(double) * ny * nz);
        check(buf != NULL);

        int n1 = ny + 2 * nghost_y;
        int n2 = nz + 2 * nghost_z;
        int iy, iz;

        for (iy = 0; iy < ny; iy++)
            for (iz = 0; iz < nz; iz++) {
                int j = iy + nghost_y;
                int k = iz + nghost_z;
                int ijk = (islice * n1 + j) * n2 + k;
                int iyiz = iy * nz + iz;
                buf[iyiz] = u[ijk];
            }

        io_write_ascii(buf, ny * nz, filename);

        free(buf);
    }
}
*/


void io_write(double ***u, char filename[],
              int nx, int ny, int nz,
              int nghost_z, int zst[], int zsz[])
{
    MPI_Datatype filetype;
    MPI_Type_create_subarray(3,
        (int []) {nx, ny, nz},
        (int []) {zsz[0], zsz[1], nz},
        (int []) {zst[0] - 1, zst[1] - 1, 0},
        MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    MPI_Datatype memtype;
    MPI_Type_create_subarray(3,
        (int []) {zsz[0], zsz[1], zsz[2]},
        (int []) {zsz[0], zsz[1], nz},
        (int []) {0, 0, nghost_z},
        MPI_ORDER_C, MPI_DOUBLE, &memtype);
    MPI_Type_commit(&memtype);

    MPI_File fh;

    // Delete previous file if it exists
    check(MPI_SUCCESS == MPI_File_open(MPI_COMM_WORLD, filename,
        MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE,
        MPI_INFO_NULL, &fh));

    check(MPI_SUCCESS == MPI_File_close(&fh));

    // Now write file
    check(MPI_SUCCESS == MPI_File_open(MPI_COMM_WORLD, filename,
        MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &fh));

    check(MPI_SUCCESS == MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype,
        "native", MPI_INFO_NULL));

    MPI_Status status;

    check(MPI_SUCCESS == MPI_File_write_all(fh, &(u[zst[0]][zst[1]][zst[2]]),
        1, memtype, &status));

    check(MPI_SUCCESS == MPI_File_close(&fh));

    MPI_Type_free(&filetype);
    MPI_Type_free(&memtype);
}


void io_read(double ***u, char filename[],
             int nx, int ny, int nz,
             int nghost_z, int zst[], int zsz[])
{
    MPI_Datatype filetype;
    MPI_Type_create_subarray(3,
        (int []) {nx, ny, nz},
        (int []) {zsz[0], zsz[1], nz},
        (int []) {zst[0] - 1, zst[1] - 1, 0},
        MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    MPI_Datatype memtype;
    MPI_Type_create_subarray(3,
        (int []) {zsz[0], zsz[1], zsz[2]},
        (int []) {zsz[0], zsz[1], nz},
        (int []) {0, 0, nghost_z},
        MPI_ORDER_C, MPI_DOUBLE, &memtype);
    MPI_Type_commit(&memtype);

    MPI_File fh;

    check(MPI_SUCCESS == MPI_File_open(MPI_COMM_WORLD, filename,
        MPI_MODE_RDONLY, MPI_INFO_NULL, &fh));

    check(MPI_SUCCESS == MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype,
        "native", MPI_INFO_NULL));

    MPI_Status status;

    check(MPI_SUCCESS == MPI_File_read_all(fh, &(u[zst[0]][zst[1]][zst[2]]),
        1, memtype, &status));

    check(MPI_SUCCESS == MPI_File_close(&fh));

    MPI_Type_free(&filetype);
    MPI_Type_free(&memtype);
}


/*
void io_write_scalarfield_vtk(double u[],
                              double x[], double y[], double z[],
                              int i0, int j0, int k0,
                              int in, int jn, int kn,
                              int n0, int n1, int n2,
                              const char dirname[],
                              const char basename[])
{
    check(0 <= i0 && i0 < in && in <= n0);
    check(0 <= j0 && j0 < jn && jn <= n1);
    check(0 <= k0 && k0 < kn && kn <= n2);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char filename[128];
    sprintf(filename, "%s/%s.%d.vtk", dirname, basename, rank);

    FILE *fp = fopen(filename, "w");
    check(fp != NULL);

    int i, j, k;
    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "%s\n", basename);
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET RECTILINEAR_GRID\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", in - i0, jn - j0, kn - k0);
    fprintf(fp, "X_COORDINATES %d float\n", in - i0);
    for (i = i0; i < in; i++) fprintf(fp, "%+e\n", x[i]);
    fprintf(fp, "Y_COORDINATES %d float\n", jn - j0);
    for (j = j0; j < jn; j++) fprintf(fp, "%+e\n", y[j]);
    fprintf(fp, "Z_COORDINATES %d float\n", kn - k0);
    for (k = k0; k < kn; k++) fprintf(fp, "%+e\n", z[k]);
    fprintf(fp, "POINT_DATA %d\n", (in - i0) * (jn - j0) * (kn - k0));
    fprintf(fp, "SCALARS %s float 1\n", basename);
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (k = k0; k < kn; k++)
        for (j = j0; j < jn; j++)
            for (i = i0; i < in; i++) {
                int ijk = (i * n1 + j) * n2 + k;
                fprintf(fp, "%+e\n", u[ijk]);
            }

    fclose(fp);
}


void io_write_vectorfield_vtk(double u[], double v[], double w[],
                              double x[], double y[], double z[],
                              int i0, int j0, int k0,
                              int in, int jn, int kn,
                              int n0, int n1, int n2,
                              const char dirname[],
                              const char basename[])
{
    check(0 <= i0 && i0 < in && in <= n0);
    check(0 <= j0 && j0 < jn && jn <= n1);
    check(0 <= k0 && k0 < kn && kn <= n2);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char filename[128];
    sprintf(filename, "%s/%s.%d.vtk", dirname, basename, rank);

    FILE *fp = fopen(filename, "w");
    check(fp != NULL);

    int i, j, k;
    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "%s\n", basename);
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET RECTILINEAR_GRID\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", in - i0, jn - j0, kn - k0);
    fprintf(fp, "X_COORDINATES %d float\n", in - i0);
    for (i = i0; i < in; i++) fprintf(fp, "%+e\n", x[i]);
    fprintf(fp, "Y_COORDINATES %d float\n", jn - j0);
    for (j = j0; j < jn; j++) fprintf(fp, "%+e\n", y[j]);
    fprintf(fp, "Z_COORDINATES %d float\n", kn - k0);
    for (k = k0; k < kn; k++) fprintf(fp, "%+e\n", z[k]);
    fprintf(fp, "POINT_DATA %d\n", (in - i0) * (jn - j0) * (kn - k0));
    fprintf(fp, "VECTORS %s float\n", basename);
    for (k = k0; k < kn; k++)
        for (j = j0; j < jn; j++)
            for (i = i0; i < in; i++) {
                int ijk = (i * n1 + j) * n2 + k;
                fprintf(fp, "%+e %+e %+e\n", u[ijk], v[ijk], w[ijk]);
            }

    fclose(fp);
}
*/
