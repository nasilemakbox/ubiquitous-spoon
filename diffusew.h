#define DIRICHW (20)

extern void apply_bc_w2(double w[],
                        int local_nx, int local_ny, int nz,
                        int nghost_x, int nghost_y, int nghost_z,
                        double xw[], double yw[], double dz[],
                        double (*bc_bot)(double, double),
                        double (*bc_top)(double, double),
                        int bctype_bot, int bctype_top);

extern void apply_bc_w4(double w[],
                        int local_nx, int local_ny, int nz,
                        int nghost_x, int nghost_y, int nghost_z,
                        double xw[], double yw[], double dz[],
                        double (*bc_bot)(double, double),
                        double (*bc_top)(double, double),
                        int bctype_bot, int bctype_top);

extern void helmholtz1d_operator_w2(double w[], double h[],
                                    int local_nx, int local_ny, int nz,
                                    int nghost_x, int nghost_y, int nghost_z,
                                    double dz[], double nudt);

extern void helmholtz1d_operator_w4(double w[], double h[],
                                    int local_nx, int local_ny, int nz,
                                    int nghost_x, int nghost_y, int nghost_z,
                                    double dz[], double nudt);

extern void solve_helmholtz1d_w2(double w[], double h[],
                                 int local_nx, int local_ny, int nz,
                                 int nghost_x, int nghost_y, int nghost_z,
                                 double dz[], double nudt,
                                 double avg_bot, double avg_top,
                                 int bctype_bot, int bctype_top);

extern void solve_helmholtz1d_w4(double w[], double h[],
                                 int local_nx, int local_ny, int nz,
                                 int nghost_x, int nghost_y, int nghost_z,
                                 double dz[], double nudt,
                                 double avg_bot, double avg_top,
                                 int bctype_bot, int bctype_top);

extern void add_laplacexy_operator_w2(double w[], double h[],
                                      int local_nx, int local_ny, int nz,
                                      int nghost_x, int nghost_y, int nghost_z,
                                      double dx, double dy, double nu);

extern void add_laplacexy_operator_w4(double w[], double h[],
                                      int local_nx, int local_ny, int nz,
                                      int nghost_x, int nghost_y, int nghost_z,
                                      double dx, double dy, double nu);

extern void helmholtz_operator_w2(double w[], double h[],
                                  int local_nx, int local_ny, int nz,
                                  int nghost_x, int nghost_y, int nghost_z,
                                  double dx, double dy, double dz[], double nudt);

extern void helmholtz_operator_w4(double w[], double h[],
                                  int local_nx, int local_ny, int nz,
                                  int nghost_x, int nghost_y, int nghost_z,
                                  double dx, double dy, double dz[], double nudt);

extern void solve_helmholtz_w2(fftw_complex cout[],
                               int nx, int local_nx, int local_x_start,
                               int ny, int local_ny, int local_y_start,
                               int nz, int nghost_z,
                               double dx, double dy, double dz[], double nudt,
                               double avg_bot, double avg_top,
                               int bctype_bot, int bctype_top);

extern void solve_helmholtz_w4(fftw_complex cout[],
                               int nx, int local_nx, int local_x_start,
                               int ny, int local_ny, int local_y_start,
                               int nz, int nghost_z,
                               double dx, double dy, double dz[], double nudt,
                               double avg_bot, double avg_top,
                               int bctype_bot, int bctype_top);

extern void diffusew_initialize2(int nz, int nghost_z);

extern void diffusew_initialize4(int nz, int nghost_z);

extern void diffusew_finalize2(void);

extern void diffusew_finalize4(void);
