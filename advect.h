extern void massfluxx2(double u[], double v[], double w[],
                       double muphx[], double mvphx[], double mwphx[],
                       int local_nx, int local_ny, int nz,
                       int nghost_x, int nghost_y, int nghost_z,
                       double dx, double dy, double dz[]);

extern void massfluxy2(double u[], double v[], double w[],
                       double muphy[], double mvphy[], double mwphy[],
                       int local_nx, int local_ny, int nz,
                       int nghost_x, int nghost_y, int nghost_z,
                       double dx, double dy, double dz[]);

extern void massfluxz2(double u[], double v[], double w[],
                       double muphz[], double mvphz[], double mwphz[],
                       int local_nx, int local_ny, int nz,
                       int nghost_x, int nghost_y, int nghost_z,
                       double dx, double dy, double dz[]);

extern void massfluxx4(double u[], double v[], double w[],
                       double muphx[], double mvphx[], double mwphx[],
                       int local_nx, int local_ny, int nz,
                       int nghost_x, int nghost_y, int nghost_z,
                       double dx, double dy, double dz[]);

extern void massfluxy4(double u[], double v[], double w[],
                       double muphy[], double mvphy[], double mwphy[],
                       int local_nx, int local_ny, int nz,
                       int nghost_x, int nghost_y, int nghost_z,
                       double dx, double dy, double dz[]);

extern void massfluxz4(double u[], double v[], double w[],
                       double muphz[], double mvphz[], double mwphz[],
                       int local_nx, int local_ny, int nz,
                       int nghost_x, int nghost_y, int nghost_z,
                       double dx, double dy, double dz[]);

extern void advectu2(double u[],
                     double muphx[], double mvphx[], double mwphx[],
                     double fu[],
                     int local_nx, int local_ny, int nz,
                     int nghost_x, int nghost_y, int nghost_z,
                     double dx, double dy, double dz[]);

extern void advectv2(double v[],
                     double muphy[], double mvphy[], double mwphy[],
                     double fv[],
                     int local_nx, int local_ny, int nz,
                     int nghost_x, int nghost_y, int nghost_z,
                     double dx, double dy, double dz[]);

extern void advectw2(double w[],
                     double muphz[], double mvphz[], double mwphz[],
                     double fw[],
                     int local_nx, int local_ny, int nz,
                     int nghost_x, int nghost_y, int nghost_z,
                     double dx, double dy, double dz[]);

extern void advectu4(double u[],
                     double muphx[], double mvphx[], double mwphx[],
                     double fu[],
                     int local_nx, int local_ny, int nz,
                     int nghost_x, int nghost_y, int nghost_z,
                     double dx, double dy, double dz[]);

extern void advectv4(double v[],
                     double muphy[], double mvphy[], double mwphy[],
                     double fv[],
                     int local_nx, int local_ny, int nz,
                     int nghost_x, int nghost_y, int nghost_z,
                     double dx, double dy, double dz[]);

extern void advectw4(double w[],
                     double muphz[], double mvphz[], double mwphz[],
                     double fw[],
                     int local_nx, int local_ny, int nz,
                     int nghost_x, int nghost_y, int nghost_z,
                     double dx, double dy, double dz[]);
