typedef struct ghostz {
    MPI_Comm *comm_cart;
    int zst[3];
    int zen[3];
    int zsz[3];
    int zst_pad[3];
    int zen_pad[3];
    int zsz_pad[3];
    int rank_x_src;
    int rank_x_dst;
    int rank_y_src;
    int rank_y_dst;
    MPI_Datatype send_x_src;
    MPI_Datatype recv_x_dst;
    MPI_Datatype send_x_dst;
    MPI_Datatype recv_x_src;
    MPI_Datatype send_y_src;
    MPI_Datatype recv_y_dst;
    MPI_Datatype send_y_dst;
    MPI_Datatype recv_y_src;
} ghostz_plan;


ghostz_plan *create_ghostz_plan(MPI_Comm *comm_cart, MPI_Datatype type,
    int zst[3], int zen[3], int zsz[3],
    int nghost_x, int nghost_y);
void destroy_ghostz_plan(ghostz_plan *p);


void ghostz_periodic(void *u_pad, ghostz_plan *p);
void ghostz_truncate_double(double ***u_pad, double ***u, ghostz_plan *p);
void ghostz_pad_double(double ***u_pad, double ***u, ghostz_plan *p);
