extern void add_buoyancy2(double c[],
                          double fu[], double fv[], double fw[],
                          double betg_x, double betg_y, double betg_z,
                          int local_nx, int local_ny, int nz,
                          int nghost_x, int nghost_y, int nghost_z);

extern void add_buoyancy4(double c[],
                          double fu[], double fv[], double fw[],
                          double betg_x, double betg_y, double betg_z,
                          int local_nx, int local_ny, int nz,
                          int nghost_x, int nghost_y, int nghost_z);

extern void advect_scalar(double mucphx[], double mvcphy[], double mwcphz[],
                          double fc[],
                          int local_nx, int local_ny, int nz,
                          int nghost_x, int nghost_y, int nghost_z,
                          double dx, double dy, double dz[]);

extern void flux_quick(double u[], double v[], double w[],
                       double c[],
                       double mucphx[], double mvcphy[], double mwcphz[],
                       int local_nx, int local_ny, int nz,
                       int nghost_x, int nghost_y, int nghost_z,
                       double dx, double dy, double dz[]);
