extern void divergence2(double u[], double v[], double w[], double d[],
                        int local_nx, int local_ny, int nz,
                        int nghost_x, int nghost_y, int nghost_z,
                        double dx, double dy, double dz[]);

extern void divergence4(double u[], double v[], double w[], double d[],
                        int local_nx, int local_ny, int nz,
                        int nghost_x, int nghost_y, int nghost_z,
                        double dx, double dy, double dz[]);

extern void project2(double u[], double v[], double w[], double p[],
                     int local_nx, int local_ny, int nz,
                     int nghost_x, int nghost_y, int nghost_z,
                     double dx, double dy, double dz[], double dt);

extern void project4(double u[], double v[], double w[], double p[],
                     int local_nx, int local_ny, int nz,
                     int nghost_x, int nghost_y, int nghost_z,
                     double dx, double dy, double dz[], double dt);

extern void pressure_laplacian2(double p[], double l[],
                                int local_nx, int local_ny, int nz,
                                int nghost_x, int nghost_y, int nghost_z,
                                double dx, double dy, double dz[], double dt);

extern void pressure_laplacian4(double p[], double l[],
                                int local_nx, int local_ny, int nz,
                                int nghost_x, int nghost_y, int nghost_z,
                                double dx, double dy, double dz[], double dt);

extern void pressure_poisson2(fftw_complex cout[],
                              int nx, int local_nx, int local_x_start,
                              int ny, int local_ny, int local_y_start,
                              int nz, int nghost_z,
                              double dx, double dy, double dz[], double dt);

extern void pressure_poisson4(fftw_complex cout[],
                              int nx, int local_nx, int local_x_start,
                              int ny, int local_ny, int local_y_start,
                              int nz, int nghost_z,
                              double dx, double dy, double dz[], double dt);

extern void pressure_initialize2(int nz, int nghost_z);

extern void pressure_initialize4(int nz, int nghost_z);

extern void pressure_finalize2(void);

extern void pressure_finalize4(void);
