#include "check.h"
#include "decomp.h"


//
// Find sub-domain information held by current processor
//   INPUT:
//     nx, ny, nz - global data dimension
//     pdim[3]    - number of processor grid in each dimension,
//                  valid values: 1 - distribute locally;
//                                2 - distribute across p_row;
//                                3 - distribute across p_col
//   OUTPUT:
//     lstart[3]  - starting index
//     lend[3]    - ending index
//     lsize[3]   - size of the sub-block (redundant)
//
void partition(int nx, int ny, int nz, int pdim[3],
    int dims[2], int coord[2],
    int lstart[3], int lend[3], int lsize[3])
{
    int *st, *en, *sz;
    int i, gsize;

    for (i = 0; i < 3; i++) {

        if (i == 0) {
            gsize = nx;
        }
        else if (i == 1) {
            gsize = ny;
        }
        else if (i == 2) {
            gsize = nz;
        }

        if (pdim[i] == 1) {         // all local
            lstart[i] = 1;
            lend[i]   = gsize;
            lsize[i]  = gsize;
        }
        else if (pdim[i] == 2) {    // distribute across dims[0]
            st = (int *) malloc(sizeof(int) * dims[0]);
            en = (int *) malloc(sizeof(int) * dims[0]);
            sz = (int *) malloc(sizeof(int) * dims[0]);
            check(st != NULL); check(en != NULL); check(sz != NULL);
            distribute(gsize, dims[0], st, en, sz);
            lstart[i] = st[coord[0]];
            lend[i]   = en[coord[0]];
            lsize[i]  = sz[coord[0]];
            free(st); free(en); free(sz);
        }
        else if (pdim[i] == 3) {    // distribute across dims[1]
            st = (int *) malloc(sizeof(int) * dims[1]);
            en = (int *) malloc(sizeof(int) * dims[1]);
            sz = (int *) malloc(sizeof(int) * dims[1]);
            check(st != NULL); check(en != NULL); check(sz != NULL);
            distribute(gsize, dims[1], st, en, sz);
            lstart[i] = st[coord[1]];
            lend[i]   = en[coord[1]];
            lsize[i]  = sz[coord[1]];
            free(st); free(en); free(sz);
        }
    }
}


//
// - distributes grid points in one dimension
// - handles uneven distribution properly
//
void distribute(int data1, int proc, int st[], int en[], int sz[])
// data1 -- data size in any dimension to be partitioned
// proc  -- number of processors in that dimension
// st    -- array of starting index
// en    -- array of ending index
// sz    -- array of local size (redundant)
{
    int i;
    int size1 = data1 / proc;
    int nu = data1 - size1 * proc;
    int nl = proc - nu;

    st[0] = 1;
    sz[0] = size1;
    en[0] = size1;
    for (i = 1; i < nl; i++) {
        st[i] = en[i - 1] + 1;
        sz[i] = size1;
        en[i] = en[i - 1] + size1;
    }
    size1 = size1 + 1;
    for (i = nl; i < proc; i++) {
        st[i] = en[i - 1] + 1;
        sz[i] = size1;
        en[i] = en[i - 1] + size1;
    }
}


//
// Define how each dimension is distributed across processors
//   e.g. 17 meshes across 4 processors would be distributed as (4,4,4,5)
//   such global information is required locally at MPI_Alltoallw time
//
void get_dist(int nx, int ny, int nz, int dims[2],
    int x1st[], int x1en[], int x1dist[],
    int y1st[], int y1en[], int y1dist[],
    int y2st[], int y2en[], int y2dist[],
    int z2st[], int z2en[], int z2dist[])
{
    distribute(nx, dims[0], x1st, x1en, x1dist);
    distribute(ny, dims[0], y1st, y1en, y1dist);

    distribute(ny, dims[1], y2st, y2en, y2dist);
    distribute(nz, dims[1], z2st, z2en, z2dist);
}


decomp_plan *create_decomp_plan(int dims[2], int coord[2],
    int nx, int ny, int nz)
{
    decomp_plan *p = (decomp_plan *) malloc(sizeof(decomp_plan));
    check(p != NULL);

    // distribute mesh points
    p->x1st   = (int *) malloc(dims[0] * sizeof(int));
    p->x1en   = (int *) malloc(dims[0] * sizeof(int));
    p->x1dist = (int *) malloc(dims[0] * sizeof(int));
    p->y1st   = (int *) malloc(dims[0] * sizeof(int));
    p->y1en   = (int *) malloc(dims[0] * sizeof(int));
    p->y1dist = (int *) malloc(dims[0] * sizeof(int));
    p->y2st   = (int *) malloc(dims[1] * sizeof(int));
    p->y2en   = (int *) malloc(dims[1] * sizeof(int));
    p->y2dist = (int *) malloc(dims[1] * sizeof(int));
    p->z2st   = (int *) malloc(dims[1] * sizeof(int));
    p->z2en   = (int *) malloc(dims[1] * sizeof(int));
    p->z2dist = (int *) malloc(dims[1] * sizeof(int));
    check(p->x1st   != NULL);
    check(p->x1en   != NULL);
    check(p->x1dist != NULL);
    check(p->y1st   != NULL);
    check(p->y1en   != NULL);
    check(p->y1dist != NULL);
    check(p->y2st   != NULL);
    check(p->y2en   != NULL);
    check(p->y2dist != NULL);
    check(p->z2st   != NULL);
    check(p->z2en   != NULL);
    check(p->z2dist != NULL);

    get_dist(nx, ny, nz, dims,
        p->x1st, p->x1en, p->x1dist,
        p->y1st, p->y1en, p->y1dist,
        p->y2st, p->y2en, p->y2dist,
        p->z2st, p->z2en, p->z2dist);

    // generate partition information - starting/ending index etc.
    partition(nx, ny, nz, (int []) {1, 2, 3}, dims, coord,
        p->xst, p->xen, p->xsz);
    partition(nx, ny, nz, (int []) {2, 1, 3}, dims, coord,
        p->yst, p->yen, p->ysz);
    partition(nx, ny, nz, (int []) {2, 3, 1}, dims, coord,
        p->zst, p->zen, p->zsz);

    return p;
}


void destroy_decomp_plan(decomp_plan *p)
{
    free(p->x1st);
    free(p->x1en);
    free(p->x1dist);
    free(p->y1st);
    free(p->y1en);
    free(p->y1dist);
    free(p->y2st);
    free(p->y2en);
    free(p->y2dist);
    free(p->z2st);
    free(p->z2en);
    free(p->z2dist);

    free(p);
}
