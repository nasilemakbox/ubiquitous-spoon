#define DIRICHU (0)
#define NEUMANU (1)

extern void apply_bc_u2(double u[],
                        int local_nx, int local_ny, int nz,
                        int nghost_x, int nghost_y, int nghost_z,
                        double xu[], double yu[], double dz[],
                        double (*bc_bot)(double, double),
                        double (*bc_top)(double, double),
                        int bctype_bot, int bctype_top);

extern void apply_bc_u4(double u[],
                        int local_nx, int local_ny, int nz,
                        int nghost_x, int nghost_y, int nghost_z,
                        double xu[], double yu[], double dz[],
                        double (*bc_bot)(double, double),
                        double (*bc_top)(double, double),
                        int bctype_bot, int bctype_top);

extern void helmholtz1d_operator_u2(double u[], double h[],
                                    int local_nx, int local_ny, int nz,
                                    int nghost_x, int nghost_y, int nghost_z,
                                    double dz[], double nudt);

extern void helmholtz1d_operator_u4(double u[], double h[],
                                    int local_nx, int local_ny, int nz,
                                    int nghost_x, int nghost_y, int nghost_z,
                                    double dz[], double nudt);

extern void solve_helmholtz1d_u2(double u[], double h[],
                                 int local_nx, int local_ny, int nz,
                                 int nghost_x, int nghost_y, int nghost_z,
                                 double dz[], double nudt,
                                 double avg_bot, double avg_top,
                                 int bctype_bot, int bctype_top);

extern void solve_helmholtz1d_u4(double u[], double h[],
                                 int local_nx, int local_ny, int nz,
                                 int nghost_x, int nghost_y, int nghost_z,
                                 double dz[], double nudt,
                                 double avg_bot, double avg_top,
                                 int bctype_bot, int bctype_top);

extern void add_laplacexy_operator_u2(double u[], double h[],
                                      int local_nx, int local_ny, int nz,
                                      int nghost_x, int nghost_y, int nghost_z,
                                      double dx, double dy, double nu);

extern void add_laplacexy_operator_u4(double u[], double h[],
                                      int local_nx, int local_ny, int nz,
                                      int nghost_x, int nghost_y, int nghost_z,
                                      double dx, double dy, double nu);

extern void helmholtz_operator_u2(double u[], double h[],
                                  int local_nx, int local_ny, int nz,
                                  int nghost_x, int nghost_y, int nghost_z,
                                  double dx, double dy, double dz[], double nudt);

extern void helmholtz_operator_u4(double u[], double h[],
                                  int local_nx, int local_ny, int nz,
                                  int nghost_x, int nghost_y, int nghost_z,
                                  double dx, double dy, double dz[], double nudt);

extern void solve_helmholtz_u2(fftw_complex cout[],
                               int nx, int local_nx, int local_x_start,
                               int ny, int local_ny, int local_y_start,
                               int nz, int nghost_z,
                               double dx, double dy, double dz[], double nudt,
                               double avg_bot, double avg_top,
                               int bctype_bot, int bctype_top);

extern void solve_helmholtz_u4(fftw_complex cout[],
                               int nx, int local_nx, int local_x_start,
                               int ny, int local_ny, int local_y_start,
                               int nz, int nghost_z,
                               double dx, double dy, double dz[], double nudt,
                               double avg_bot, double avg_top,
                               int bctype_bot, int bctype_top);

extern void diffuseu_initialize2(int nz, int nghost_z);

extern void diffuseu_initialize4(int nz, int nghost_z);

extern void diffuseu_finalize2(void);

extern void diffuseu_finalize4(void);
