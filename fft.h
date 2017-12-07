typedef struct fftx {
    int nx;
    int xst_d[3];
    int xen_d[3];
    int xsz_d[3];
    int xst_c[3];
    int xen_c[3];
    int xsz_c[3];
    double ***work_d;
    fftw_complex ***work_c;
    fftw_plan d2c;
    fftw_plan c2d;
} fftx_plan;

typedef struct ffty {
    int ny;
    int yst_c[3];
    int yen_c[3];
    int ysz_c[3];
    fftw_complex ***work_i;
    fftw_complex ***work_o;
    fftw_plan i2o;
    fftw_plan o2i;
} ffty_plan;


fftx_plan *create_fftx_plan(int nx,
    int xst_d[3], int xen_d[3], int xsz_d[3],
    int xst_c[3], int xen_c[3], int xsz_c[3]);
ffty_plan *create_ffty_plan(int ny,
    int yst_c[3], int yen_c[3], int ysz_c[3]);
void destroy_fftx_plan(fftx_plan *p);
void destroy_ffty_plan(ffty_plan *p);


void fftx_d2c(fftx_plan *p);
void fftx_c2d(fftx_plan *p);
void ffty_i2o(ffty_plan *p);
void ffty_o2i(ffty_plan *p);
