extern void io_append_ascii(double u[], int n, char filename[]);

extern void io_write_ascii(double u[], int n, char filename[]);

/*
extern void io_write_kslice(double u[], char filename[],
                            int local_nx, int ny, int nz,
                            int nghost_x, int nghost_y, int nghost_z,
                            int local_x_start, int kslice);

extern void io_write_jslice(double u[], char filename[],
                            int local_nx, int ny, int nz,
                            int nghost_x, int nghost_y, int nghost_z,
                            int local_x_start, int jslice);

extern void io_write_islice(double u[], char filename[],
                            int local_nx, int ny, int nz,
                            int nghost_y, int nghost_z,
                            int local_x_start, int islice);
*/

extern void io_write(double ***u, char filename[],
                     int nx, int ny, int nz,
                     int nghost_z, int zst[], int zsz[]);


extern void io_read(double ***u, char filename[],
                    int nx, int ny, int nz,
                    int nghost_z, int zst[], int zsz[]);

/*
extern void io_write_scalarfield_vtk(double u[],
                                     double x[], double y[], double z[],
                                     int i0, int j0, int k0,
                                     int in, int jn, int kn,
                                     int n0, int n1, int n2,
                                     const char dirname[],
                                     const char basename[]);

extern void io_write_vectorfield_vtk(double u[], double v[], double w[],
                                     double x[], double y[], double z[],
                                     int i0, int j0, int k0,
                                     int in, int jn, int kn,
                                     int n0, int n1, int n2,
                                     const char dirname[],
                                     const char basename[]);
*/
