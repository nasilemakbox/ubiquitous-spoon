void partition(int nx, int ny, int nz, int pdim[3],
    int dims[2], int coord[2],
    int lstart[3], int lend[3], int lsize[3]);
void distribute(int data1, int proc, int st[], int en[], int sz[]);
void get_dist(int nx, int ny, int nz, int dims[2],
    int x1st[], int x1en[], int x1dist[],
    int y1st[], int y1en[], int y1dist[],
    int y2st[], int y2en[], int y2dist[],
    int z2st[], int z2en[], int z2dist[]);


typedef struct decomp {
    int xst[3];
    int xen[3];
    int xsz[3];
    int yst[3];
    int yen[3];
    int ysz[3];
    int zst[3];
    int zen[3];
    int zsz[3];
    int *x1st;
    int *x1en;
    int *x1dist;
    int *y1st;
    int *y1en;
    int *y1dist;
    int *y2st;
    int *y2en;
    int *y2dist;
    int *z2st;
    int *z2en;
    int *z2dist;
} decomp_plan;


decomp_plan *create_decomp_plan(int dims[2], int coord[2],
    int nx, int ny, int nz);
void destroy_decomp_plan(decomp_plan *p);
