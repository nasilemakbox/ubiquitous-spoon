extern void get_xy_spec(fftw_complex uhat[], fftw_complex vhat[],
                        double specx[], double specy[],
                        int nx, int local_nx, int local_x_start,
                        int ny, int local_ny, int local_y_start,
                        int nz, int nghost_z);

extern void get_xy_moment(double u[], double moment[], double en,
                          int local_nx, int local_ny, int nz,
                          int nghost_x, int nghost_y, int nghost_z,
                          int nx, int ny);

extern void get_xy_corr(double u[], double v[], double corr[],
                        int local_nx, int local_ny, int nz,
                        int nghost_x, int nghost_y, int nghost_z,
                        int nx, int ny);

/*
extern void get_y_corr(double u[], double v[], double corr[],
                       int local_nx, int ny, int nz,
                       int nghost_x, int nghost_y, int nghost_z);
*/

extern void get_x_corr(double u[], double v[], double corr[],
                       int local_nx, int local_ny, int nz,
                       int local_y_start,
                       int nghost_x, int nghost_y, int nghost_z,
                       int nx, int ny);

extern void get_xy_avg(double u[], double avg[],
                       int local_nx, int local_ny, int nz,
                       int nghost_x, int nghost_y, int nghost_z,
                       int nx, int ny);

/*
extern void get_y_avg(double u[], double avg[],
                      int local_nx, int ny, int nz,
                      int nghost_x, int nghost_y, int nghost_z);
*/

extern void get_x_avg(double u[], double avg[],
                      int local_nx, int local_ny, int nz,
                      int local_y_start,
                      int nghost_x, int nghost_y, int nghost_z,
                      int nx, int ny);

extern double get_cfl(double u[], double v[], double w[],
                      int local_nx, int local_ny, int nz,
                      int nghost_x, int nghost_y, int nghost_z,
                      double dx, double dy, double dz[], double dt);

extern void get_avg_blowing_z(double u[],
                              int local_nx, int local_ny, int nz,
                              int nghost_x, int nghost_y, int nghost_z,
                              int nx, int ny, double *avg_bot, double *avg_top);

extern void get_z_at_p(double dz[], double z[], int nz, int nghost_z);

extern void get_z_at_w(double dz[], double z[], int nz, int nghost_z);

extern double get_z_avg(double u[], int nz, int nghost_z, double dz[]);
