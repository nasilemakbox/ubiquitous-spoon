#include "check.h"
#include "ghostz.h"


ghostz_plan *create_ghostz_plan(MPI_Comm *comm_cart, MPI_Datatype type,
    int zst[3], int zen[3], int zsz[3],
    int nghost_x, int nghost_y)
{
    check(zsz[0] >= 2 * nghost_x);
    check(zsz[1] >= 2 * nghost_y);

    ghostz_plan *p = (ghostz_plan *) malloc(sizeof(ghostz_plan));
    check(p != NULL);

    p->comm_cart = comm_cart;

    p->zst[0] = zst[0];
    p->zst[1] = zst[1];
    p->zst[2] = zst[2];
    p->zen[0] = zen[0];
    p->zen[1] = zen[1];
    p->zen[2] = zen[2];
    p->zsz[0] = zsz[0];
    p->zsz[1] = zsz[1];
    p->zsz[2] = zsz[2];

    p->zst_pad[0] = zst[0] - nghost_x;
    p->zst_pad[1] = zst[1] - nghost_y;
    p->zst_pad[2] = zst[2];
    p->zen_pad[0] = zen[0] + nghost_x;
    p->zen_pad[1] = zen[1] + nghost_y;
    p->zen_pad[2] = zen[2];
    p->zsz_pad[0] = zsz[0] + 2 * nghost_x;
    p->zsz_pad[1] = zsz[1] + 2 * nghost_y;
    p->zsz_pad[2] = zsz[2];

    MPI_Cart_shift(*comm_cart, 0, // direction
        1, // displacement
        &(p->rank_x_src), &(p->rank_x_dst));
    MPI_Cart_shift(*comm_cart, 1, // direction
        1, // displacement
        &(p->rank_y_src), &(p->rank_y_dst));

    MPI_Type_create_subarray(3, p->zsz_pad,
        (int []) {nghost_x, p->zsz_pad[1], p->zsz_pad[2]},
        (int []) {nghost_x, 0, 0},
        MPI_ORDER_C, type, &(p->send_x_src));
    MPI_Type_create_subarray(3, p->zsz_pad,
        (int []) {nghost_x, p->zsz_pad[1], p->zsz_pad[2]},
        (int []) {p->zsz_pad[0] - nghost_x, 0, 0},
        MPI_ORDER_C, type, &(p->recv_x_dst));
    MPI_Type_commit(&(p->send_x_src));
    MPI_Type_commit(&(p->recv_x_dst));

    MPI_Type_create_subarray(3, p->zsz_pad,
        (int []) {nghost_x, p->zsz_pad[1], p->zsz_pad[2]},
        (int []) {p->zsz_pad[0] - 2 * nghost_x, 0, 0},
        MPI_ORDER_C, type, &(p->send_x_dst));
    MPI_Type_create_subarray(3, p->zsz_pad,
        (int []) {nghost_x, p->zsz_pad[1], p->zsz_pad[2]},
        (int []) {0, 0, 0},
        MPI_ORDER_C, type, &(p->recv_x_src));
    MPI_Type_commit(&(p->send_x_dst));
    MPI_Type_commit(&(p->recv_x_src));

    MPI_Type_create_subarray(3, p->zsz_pad,
        (int []) {p->zsz_pad[0], nghost_y, p->zsz_pad[2]},
        (int []) {0, nghost_y, 0},
        MPI_ORDER_C, type, &(p->send_y_src));
    MPI_Type_create_subarray(3, p->zsz_pad,
        (int []) {p->zsz_pad[0], nghost_y, p->zsz_pad[2]},
        (int []) {0, p->zsz_pad[1] - nghost_y, 0},
        MPI_ORDER_C, type, &(p->recv_y_dst));
    MPI_Type_commit(&(p->send_y_src));
    MPI_Type_commit(&(p->recv_y_dst));

    MPI_Type_create_subarray(3, p->zsz_pad,
        (int []) {p->zsz_pad[0], nghost_y, p->zsz_pad[2]},
        (int []) {0, p->zsz_pad[1] - 2 * nghost_y, 0},
        MPI_ORDER_C, type, &(p->send_y_dst));
    MPI_Type_create_subarray(3, p->zsz_pad,
        (int []) {p->zsz_pad[0], nghost_y, p->zsz_pad[2]},
        (int []) {0, 0, 0},
        MPI_ORDER_C, type, &(p->recv_y_src));
    MPI_Type_commit(&(p->send_y_dst));
    MPI_Type_commit(&(p->recv_y_src));

    return p;
}


void destroy_ghostz_plan(ghostz_plan *p)
{
    MPI_Type_free(&(p->send_x_src));
    MPI_Type_free(&(p->recv_x_dst));
    MPI_Type_free(&(p->send_x_dst));
    MPI_Type_free(&(p->recv_x_src));
    MPI_Type_free(&(p->send_y_src));
    MPI_Type_free(&(p->recv_y_dst));
    MPI_Type_free(&(p->send_y_dst));
    MPI_Type_free(&(p->recv_y_src));

    free(p);
}


void ghostz_periodic(void *u_pad, ghostz_plan *p)
{
    int nrank;
    MPI_Comm_rank(*(p->comm_cart), &nrank);

    MPI_Request request0, request1;
    MPI_Status status;
    int tag_send, tag_recv;

    tag_send = nrank;
    tag_recv = p->rank_x_dst;
    MPI_Isend(u_pad,
        1, p->send_x_src, p->rank_x_src, tag_send, *(p->comm_cart), &request0);
    MPI_Recv(u_pad,
        1, p->recv_x_dst, p->rank_x_dst, tag_recv, *(p->comm_cart), &status);

    tag_send = nrank;
    tag_recv = p->rank_x_src;
    MPI_Isend(u_pad,
        1, p->send_x_dst, p->rank_x_dst, tag_send, *(p->comm_cart), &request1);
    MPI_Recv(u_pad,
        1, p->recv_x_src, p->rank_x_src, tag_recv, *(p->comm_cart), &status);

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);

    tag_send = nrank;
    tag_recv = p->rank_y_dst;
    MPI_Isend(u_pad,
        1, p->send_y_src, p->rank_y_src, tag_send, *(p->comm_cart), &request0);
    MPI_Recv(u_pad,
        1, p->recv_y_dst, p->rank_y_dst, tag_recv, *(p->comm_cart), &status);

    tag_send = nrank;
    tag_recv = p->rank_y_src;
    MPI_Isend(u_pad,
        1, p->send_y_dst, p->rank_y_dst, tag_send, *(p->comm_cart), &request1);
    MPI_Recv(u_pad,
        1, p->recv_y_src, p->rank_y_src, tag_recv, *(p->comm_cart), &status);

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);
}


void ghostz_truncate_double(double ***u_pad, double ***u, ghostz_plan *p)
{
    int i, j, k;

    for (i = p->zst[0]; i <= p->zen[0]; i++) {
        for (j = p->zst[1]; j <= p->zen[1]; j++) {
            for (k = p->zst[2]; k <= p->zen[2]; k++) {
                u[i][j][k] = u_pad[i][j][k];
            }
        }
    }
}


void ghostz_pad_double(double ***u_pad, double ***u, ghostz_plan *p)
{
    int i, j, k;

    for (i = p->zst[0]; i <= p->zen[0]; i++) {
        for (j = p->zst[1]; j <= p->zen[1]; j++) {
            for (k = p->zst[2]; k <= p->zen[2]; k++) {
                u_pad[i][j][k] = u[i][j][k];
            }
        }
    }
}
