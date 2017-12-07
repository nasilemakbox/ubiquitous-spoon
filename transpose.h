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
    int y2st[], int y2en[]);
void prepare_buffer_yx(int dims[2],
    int ydispls_yx[], int xdispls_yx[],
    int ycnts_yx[], int xcnts_yx[],
    MPI_Datatype ytypes_yx[], MPI_Datatype xtypes_yx[],
    MPI_Comm *comm_cart, MPI_Datatype type,
    int yst[], int yen[], int ysz[],
    int xst[], int xen[], int xsz[],
    int y1st[], int y1dist[],
    int x1st[], int x1dist[],
    int z2st[], int z2en[]);
void prepare_buffer_zy(int dims[2],
    int zdispls_zy[], int ydispls_zy[],
    int zcnts_zy[], int ycnts_zy[],
    MPI_Datatype ztypes_zy[], MPI_Datatype ytypes_zy[],
    MPI_Comm *comm_cart, MPI_Datatype type,
    int zst[], int zen[], int zsz[],
    int yst[], int yen[], int ysz[],
    int z2st[], int z2dist[],
    int y2st[], int y2dist[],
    int x1st[], int x1en[]);


typedef struct transpose {
    MPI_Comm *comm_cart;
    int *xdispls_xz;
    int *zdispls_xz;
    int *xcnts_xz;
    int *zcnts_xz;
    MPI_Datatype *xtypes_xz;
    MPI_Datatype *ztypes_xz;
    int *ydispls_yx;
    int *xdispls_yx;
    int *ycnts_yx;
    int *xcnts_yx;
    MPI_Datatype *ytypes_yx;
    MPI_Datatype *xtypes_yx;
    int *zdispls_zy;
    int *ydispls_zy;
    int *zcnts_zy;
    int *ycnts_zy;
    MPI_Datatype *ztypes_zy;
    MPI_Datatype *ytypes_zy;
} transpose_plan;


transpose_plan *create_transpose_plan(int dims[2],
    MPI_Comm *comm_cart, MPI_Datatype type,
    int xst[], int xen[], int xsz[],
    int yst[], int yen[], int ysz[],
    int zst[], int zen[], int zsz[],
    int x1st[], int x1en[], int x1dist[],
    int y1st[], int y1en[], int y1dist[],
    int y2st[], int y2en[], int y2dist[],
    int z2st[], int z2en[], int z2dist[]);
void destroy_transpose_plan(transpose_plan *p);


void transpose_z_to_y(void *src, void *dst, transpose_plan *p);
void transpose_y_to_z(void *src, void *dst, transpose_plan *p);
void transpose_y_to_x(void *src, void *dst, transpose_plan *p);
void transpose_x_to_y(void *src, void *dst, transpose_plan *p);
void transpose_x_to_z(void *src, void *dst, transpose_plan *p);
void transpose_z_to_x(void *src, void *dst, transpose_plan *p);
