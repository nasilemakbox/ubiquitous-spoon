#include "check.h"
#include "transpose.h"


//
// Prepare the send / receive buffers for MPI_Alltoallw communications
//
void prepare_buffer_xz(int dims[2],
    int xdispls_xz[], int zdispls_xz[],
    int xcnts_xz[], int zcnts_xz[],
    MPI_Datatype xtypes_xz[], MPI_Datatype ztypes_xz[],
    MPI_Comm *comm_cart, MPI_Datatype type,
    int xst[], int xen[], int xsz[],
    int zst[], int zen[], int zsz[],
    int x1st[], int x1dist[],
    int z2st[], int z2dist[],
    int y1st[], int y1en[],
    int y2st[], int y2en[])
{
    int i, k;
    int rank_x, rank_z;
    int subsize_y, offset_y;

    // Information for MPI_Alltoallw for X <=> Z transposes
    for (k = 0; k < dims[0]; k++) {
        for (i = 0; i < dims[1]; i++) {

            // Actually, rank_x and rank_z are the same.
            MPI_Cart_rank(*comm_cart, (int []) {k, i}, &rank_x);
            rank_z = rank_x;

            zdispls_xz[rank_z] = 0;
            xdispls_xz[rank_x] = 0;

            if (zst[1] <= y1en[k] && zen[1] >= y1st[k]
                && zsz[0] > 0 && z2dist[i] > 0) {

                // send to process with x-pencil defined by (k,i)
                // x-bounds are taken from the z-pencils.
                //                 zst[0]:zen[0]
                // y-bounds are the overlapping region of both pencils.
                //                 max(zst[1],y1st[k]):min(zen[1],y1en[k])
                // z-bounds are taken from the x-pencils.
                //                 z2st[i]:z2en[i]
                zcnts_xz[rank_z] = 1;
                subsize_y = MIN(zen[1], y1en[k]) - MAX(zst[1], y1st[k]) + 1;
                offset_y = MAX(zst[1], y1st[k]) - zst[1];
                MPI_Type_create_subarray(3, zsz,
                    (int []) {zsz[0], subsize_y, z2dist[i]},
                    (int []) {0, offset_y, z2st[i] - zst[2]},
                    MPI_ORDER_C, type, &ztypes_xz[rank_z]);
                MPI_Type_commit(&ztypes_xz[rank_z]);
            }
            else {
                zcnts_xz[rank_z] = 0;
                MPI_Type_create_subarray(3, (int []) {1, 1, 1},
                    (int []) {1, 1, 1},
                    (int []) {0, 0, 0},
                    MPI_ORDER_C, type, &ztypes_xz[rank_z]);
                MPI_Type_commit(&ztypes_xz[rank_z]);
            }
            if (xst[1] <= y2en[i] && xen[1] >= y2st[i]
                && x1dist[k] > 0 && xsz[2] > 0) {

                // recv from process with z-pencil defined by (k,i)
                // x-bounds are taken from the z-pencils.
                //                 x1st[k]:x1en[k]
                // y-bounds are the overlapping region of both pencils.
                //                 max(xst[1],y2st[i]):min(xen[1],y2en[i])
                // z-bounds are taken from the x-pencils.
                //                 xst[2]:xen[2]
                xcnts_xz[rank_x] = 1;
                subsize_y = MIN(xen[1], y2en[i]) - MAX(xst[1], y2st[i]) + 1;
                offset_y = MAX(xst[1], y2st[i]) - xst[1];
                MPI_Type_create_subarray(3, xsz,
                    (int []) {x1dist[k], subsize_y, xsz[2]},
                    (int []) {x1st[k] - xst[0], offset_y, 0},
                    MPI_ORDER_C, type, &xtypes_xz[rank_x]);
                MPI_Type_commit(&xtypes_xz[rank_x]);
            }
            else {
                xcnts_xz[rank_x] = 0;
                MPI_Type_create_subarray(3, (int []) {1, 1, 1},
                    (int []) {1, 1, 1},
                    (int []) {0, 0, 0},
                    MPI_ORDER_C, type, &xtypes_xz[rank_x]);
                MPI_Type_commit(&xtypes_xz[rank_x]);
            }
        }
    }
}


//
// Prepare the send / receive buffers for MPI_Alltoallw communications
//
void prepare_buffer_yx(int dims[2],
    int ydispls_yx[], int xdispls_yx[],
    int ycnts_yx[], int xcnts_yx[],
    MPI_Datatype ytypes_yx[], MPI_Datatype xtypes_yx[],
    MPI_Comm *comm_cart, MPI_Datatype type,
    int yst[], int yen[], int ysz[],
    int xst[], int xen[], int xsz[],
    int y1st[], int y1dist[],
    int x1st[], int x1dist[],
    int z2st[], int z2en[])
{
    int i, k;
    int rank_y, rank_x;
    int subsize_z, offset_z;

    // Information for MPI_Alltoallw for Y <=> X transposes
    for (k = 0; k < dims[0]; k++) {
        for (i = 0; i < dims[1]; i++) {

            // Actually, rank_y and rank_x are the same.
            MPI_Cart_rank(*comm_cart, (int []) {k, i}, &rank_y);
            rank_x = rank_y;

            xdispls_yx[rank_x] = 0;
            ydispls_yx[rank_y] = 0;

            if (xst[2] <= z2en[i] && xen[2] >= z2st[i]
                && xsz[1] > 0 && x1dist[k] > 0) {

                // send to process with y-pencil defined by (k,i)
                // x-bounds are taken from the y-pencils.
                //                 x1st[k]:x1en[k]
                // y-bounds are taken from the x-pencils.
                //                 xst[1]:xen[1]
                // z-bounds are the overlapping region of both pencils.
                //                 max(xst[2],z2st[i]):min(xen[2],z2en[i])
                xcnts_yx[rank_x] = 1;
                subsize_z = MIN(xen[2], z2en[i]) - MAX(xst[2], z2st[i]) + 1;
                offset_z = MAX(xst[2], z2st[i]) - xst[2];
                MPI_Type_create_subarray(3, xsz,
                    (int []) {x1dist[k], xsz[1], subsize_z},
                    (int []) {x1st[k] - xst[0], 0, offset_z},
                    MPI_ORDER_C, type, &xtypes_yx[rank_x]);
                MPI_Type_commit(&xtypes_yx[rank_x]);
            }
            else {
                xcnts_yx[rank_x] = 0;
                MPI_Type_create_subarray(3, (int []) {1, 1, 1},
                    (int []) {1, 1, 1},
                    (int []) {0, 0, 0},
                    MPI_ORDER_C, type, &xtypes_yx[rank_x]);
                MPI_Type_commit(&xtypes_yx[rank_x]);
            }
            if (yst[2] <= z2en[i] && yen[2] >= z2st[i]
                && y1dist[k] > 0 && ysz[0] > 0) {

                // recv from process with x-pencil defined by (k,i)
                // x-bounds are taken from the y-pencils.
                //                 yst[0]:yen[0]
                // y-bounds are taken from the x-pencils.
                //                 y1st[k]:y1en[k]
                // z-bounds are the overlapping region of both pencils.
                //                 max(yst[2],z2st[i]):min(yen[2],z2en[i])
                ycnts_yx[rank_y] = 1;
                subsize_z = MIN(yen[2], z2en[i]) - MAX(yst[2], z2st[i]) + 1;
                offset_z = MAX(yst[2], z2st[i]) - yst[2];
                MPI_Type_create_subarray(3, ysz,
                    (int []) {ysz[0], y1dist[k], subsize_z},
                    (int []) {0, y1st[k] - yst[1], offset_z},
                    MPI_ORDER_C, type, &ytypes_yx[rank_y]);
                MPI_Type_commit(&ytypes_yx[rank_y]);
            }
            else {
                ycnts_yx[rank_y] = 0;
                MPI_Type_create_subarray(3, (int []) {1, 1, 1},
                    (int []) {1, 1, 1},
                    (int []) {0, 0, 0},
                    MPI_ORDER_C, type, &ytypes_yx[rank_y]);
                MPI_Type_commit(&ytypes_yx[rank_y]);
            }
        }
    }
}


//
// Prepare the send / receive buffers for MPI_Alltoallw communications
//
void prepare_buffer_zy(int dims[2],
    int zdispls_zy[], int ydispls_zy[],
    int zcnts_zy[], int ycnts_zy[],
    MPI_Datatype ztypes_zy[], MPI_Datatype ytypes_zy[],
    MPI_Comm *comm_cart, MPI_Datatype type,
    int zst[], int zen[], int zsz[],
    int yst[], int yen[], int ysz[],
    int z2st[], int z2dist[],
    int y2st[], int y2dist[],
    int x1st[], int x1en[])
{
    int i, k;
    int rank_z, rank_y;
    int subsize_x, offset_x;

    // Information for MPI_Alltoallw for Z <=> Y transposes
    for (k = 0; k < dims[0]; k++) {
        for (i = 0; i < dims[1]; i++) {

            // Actually, rank_z and rank_y are the same.
            MPI_Cart_rank(*comm_cart, (int []) {k, i}, &rank_z);
            rank_y = rank_z;

            ydispls_zy[rank_y] = 0;
            zdispls_zy[rank_z] = 0;

            if (yst[0] <= x1en[k] && yen[0] >= x1st[k]
                && ysz[2] > 0 && y2dist[i] > 0) {

                // send to process with z-pencil defined by (k,i)
                // x-bounds are the overlapping region of both pencils.
                //                 max(yst[0],x1st[k]):min(yen[0],x1en[k])
                // y-bounds are taken from the z-pencils.
                //                 y2st[i]:y2en[i]
                // z-bounds are taken from the y-pencils.
                //                 yst[2]:yen[2]
                ycnts_zy[rank_y] = 1;
                subsize_x = MIN(yen[0], x1en[k]) - MAX(yst[0], x1st[k]) + 1;
                offset_x = MAX(yst[0], x1st[k]) - yst[0];
                MPI_Type_create_subarray(3, ysz,
                    (int []) {subsize_x, y2dist[i], ysz[2]},
                    (int []) {offset_x, y2st[i] - yst[1], 0},
                    MPI_ORDER_C, type, &ytypes_zy[rank_y]);
                MPI_Type_commit(&ytypes_zy[rank_y]);
            }
            else {
                ycnts_zy[rank_y] = 0;
                MPI_Type_create_subarray(3, (int []) {1, 1, 1},
                    (int []) {1, 1, 1},
                    (int []) {0, 0, 0},
                    MPI_ORDER_C, type, &ytypes_zy[rank_y]);
                MPI_Type_commit(&ytypes_zy[rank_y]);
            }
            if (zst[0] <= x1en[k] && zen[0] >= x1st[k]
                && z2dist[i] > 0 && zsz[1] > 0) {

                // recv from process with y-pencil defined by (k,i)
                // x-bounds are the overlapping region of both pencils.
                //                 max(zst[0],x1st[k]):min(zen[0],x1en[k])
                // y-bounds are taken from the z-pencils.
                //                 zst[1]:zen[1]
                // z-bounds are taken from the y-pencils.
                //                 z2st[i]:z2en[i]
                zcnts_zy[rank_z] = 1;
                subsize_x = MIN(zen[0], x1en[k]) - MAX(zst[0], x1st[k]) + 1;
                offset_x = MAX(zst[0], x1st[k]) - zst[0];
                MPI_Type_create_subarray(3, zsz,
                    (int []) {subsize_x, zsz[1], z2dist[i]},
                    (int []) {offset_x, 0, z2st[i] - zst[2]},
                    MPI_ORDER_C, type, &ztypes_zy[rank_z]);
                MPI_Type_commit(&ztypes_zy[rank_z]);
            }
            else {
                zcnts_zy[rank_z] = 0;
                MPI_Type_create_subarray(3, (int []) {1, 1, 1},
                    (int []) {1, 1, 1},
                    (int []) {0, 0, 0},
                    MPI_ORDER_C, type, &ztypes_zy[rank_z]);
                MPI_Type_commit(&ztypes_zy[rank_z]);
            }
        }
    }
}


transpose_plan *create_transpose_plan(int dims[2],
    MPI_Comm *comm_cart, MPI_Datatype type,
    int xst[], int xen[], int xsz[],
    int yst[], int yen[], int ysz[],
    int zst[], int zen[], int zsz[],
    int x1st[], int x1en[], int x1dist[],
    int y1st[], int y1en[], int y1dist[],
    int y2st[], int y2en[], int y2dist[],
    int z2st[], int z2en[], int z2dist[])
{
    transpose_plan *p = (transpose_plan *) malloc(sizeof(transpose_plan));
    check(p != NULL);

    p->comm_cart = comm_cart;

    int nproc;
    MPI_Comm_size(*(p->comm_cart), &nproc);

    check(nproc == dims[0] * dims[1]);

    p->xdispls_xz = (int *) malloc(nproc * sizeof(int));
    p->zdispls_xz = (int *) malloc(nproc * sizeof(int));
    p->xcnts_xz   = (int *) malloc(nproc * sizeof(int));
    p->zcnts_xz   = (int *) malloc(nproc * sizeof(int));
    p->xtypes_xz = (MPI_Datatype *) malloc(nproc * sizeof(MPI_Datatype));
    p->ztypes_xz = (MPI_Datatype *) malloc(nproc * sizeof(MPI_Datatype));
    check(p->xdispls_xz != NULL);
    check(p->zdispls_xz != NULL);
    check(p->xcnts_xz != NULL);
    check(p->zcnts_xz != NULL);
    check(p->xtypes_xz != NULL);
    check(p->ztypes_xz != NULL);

    prepare_buffer_xz(dims,
        p->xdispls_xz, p->zdispls_xz,
        p->xcnts_xz, p->zcnts_xz,
        p->xtypes_xz, p->ztypes_xz,
        p->comm_cart, type,
        xst, xen, xsz,
        zst, zen, zsz,
        x1st, x1dist,
        z2st, z2dist,
        y1st, y1en,
        y2st, y2en);

    p->ydispls_yx = (int *) malloc(nproc * sizeof(int));
    p->xdispls_yx = (int *) malloc(nproc * sizeof(int));
    p->ycnts_yx   = (int *) malloc(nproc * sizeof(int));
    p->xcnts_yx   = (int *) malloc(nproc * sizeof(int));
    p->ytypes_yx = (MPI_Datatype *) malloc(nproc * sizeof(MPI_Datatype));
    p->xtypes_yx = (MPI_Datatype *) malloc(nproc * sizeof(MPI_Datatype));
    check(p->ydispls_yx != NULL);
    check(p->xdispls_yx != NULL);
    check(p->ycnts_yx != NULL);
    check(p->xcnts_yx != NULL);
    check(p->ytypes_yx != NULL);
    check(p->xtypes_yx != NULL);

    prepare_buffer_yx(dims,
        p->ydispls_yx, p->xdispls_yx,
        p->ycnts_yx, p->xcnts_yx,
        p->ytypes_yx, p->xtypes_yx,
        p->comm_cart, type,
        yst, yen, ysz,
        xst, xen, xsz,
        y1st, y1dist,
        x1st, x1dist,
        z2st, z2en);

    p->zdispls_zy = (int *) malloc(nproc * sizeof(int));
    p->ydispls_zy = (int *) malloc(nproc * sizeof(int));
    p->zcnts_zy   = (int *) malloc(nproc * sizeof(int));
    p->ycnts_zy   = (int *) malloc(nproc * sizeof(int));
    p->ztypes_zy = (MPI_Datatype *) malloc(nproc * sizeof(MPI_Datatype));
    p->ytypes_zy = (MPI_Datatype *) malloc(nproc * sizeof(MPI_Datatype));
    check(p->zdispls_zy != NULL);
    check(p->ydispls_zy != NULL);
    check(p->zcnts_zy != NULL);
    check(p->ycnts_zy != NULL);
    check(p->ztypes_zy != NULL);
    check(p->ytypes_zy != NULL);

    prepare_buffer_zy(dims,
        p->zdispls_zy, p->ydispls_zy,
        p->zcnts_zy, p->ycnts_zy,
        p->ztypes_zy, p->ytypes_zy,
        p->comm_cart, type,
        zst, zen, zsz,
        yst, yen, ysz,
        z2st, z2dist,
        y2st, y2dist,
        x1st, x1en);

    return p;
}


void destroy_transpose_plan(transpose_plan *p)
{
    int iproc, nproc;
    MPI_Comm_size(*(p->comm_cart), &nproc);

    free(p->xdispls_xz);
    free(p->zdispls_xz);
    free(p->xcnts_xz);
    free(p->zcnts_xz);
    for (iproc = 0; iproc < nproc; iproc++) {
        MPI_Type_free(&(p->xtypes_xz[iproc]));
        MPI_Type_free(&(p->ztypes_xz[iproc]));
    }
    free(p->xtypes_xz);
    free(p->ztypes_xz);

    free(p->ydispls_yx);
    free(p->xdispls_yx);
    free(p->ycnts_yx);
    free(p->xcnts_yx);
    for (iproc = 0; iproc < nproc; iproc++) {
        MPI_Type_free(&(p->ytypes_yx[iproc]));
        MPI_Type_free(&(p->xtypes_yx[iproc]));
    }
    free(p->ytypes_yx);
    free(p->xtypes_yx);

    free(p->zdispls_zy);
    free(p->ydispls_zy);
    free(p->zcnts_zy);
    free(p->ycnts_zy);
    for (iproc = 0; iproc < nproc; iproc++) {
        MPI_Type_free(&(p->ztypes_zy[iproc]));
        MPI_Type_free(&(p->ytypes_zy[iproc]));
    }
    free(p->ztypes_zy);
    free(p->ytypes_zy);

    free(p);
}


void transpose_z_to_y(void *src, void *dst, transpose_plan *p)
{
    MPI_Alltoallw(src, p->zcnts_zy, p->zdispls_zy, p->ztypes_zy,
        dst, p->ycnts_zy, p->ydispls_zy, p->ytypes_zy, *(p->comm_cart));
}


void transpose_y_to_z(void *src, void *dst, transpose_plan *p)
{
    MPI_Alltoallw(src, p->ycnts_zy, p->ydispls_zy, p->ytypes_zy,
        dst, p->zcnts_zy, p->zdispls_zy, p->ztypes_zy, *(p->comm_cart));
}


void transpose_y_to_x(void *src, void *dst, transpose_plan *p)
{
    MPI_Alltoallw(src, p->ycnts_yx, p->ydispls_yx, p->ytypes_yx,
        dst, p->xcnts_yx, p->xdispls_yx, p->xtypes_yx, *(p->comm_cart));
}


void transpose_x_to_y(void *src, void *dst, transpose_plan *p)
{
    MPI_Alltoallw(src, p->xcnts_yx, p->xdispls_yx, p->xtypes_yx,
        dst, p->ycnts_yx, p->ydispls_yx, p->ytypes_yx, *(p->comm_cart));
}


void transpose_x_to_z(void *src, void *dst, transpose_plan *p)
{
    MPI_Alltoallw(src, p->xcnts_xz, p->xdispls_xz, p->xtypes_xz,
        dst, p->zcnts_xz, p->zdispls_xz, p->ztypes_xz, *(p->comm_cart));
}


void transpose_z_to_x(void *src, void *dst, transpose_plan *p)
{
    MPI_Alltoallw(src, p->zcnts_xz, p->zdispls_xz, p->ztypes_xz,
        dst, p->xcnts_xz, p->xdispls_xz, p->xtypes_xz, *(p->comm_cart));
}
