#include "check.h"
#include "tridag.h"
 
void tridag(double a[], double b[], double c[], double r[], double u[],
            int n)
{
    int j;
    double bet;
    double *gam = (double *) malloc(sizeof(double) * n);

    check(gam != NULL);

    check(b[0] != 0.0);
    u[0] = r[0] / (bet = b[0]);
    for (j = 1; j < n; j++) {
        gam[j] = c[j - 1] / bet;
        bet = b[j] - a[j] * gam[j];
        check(bet != 0.0);
        u[j] = (r[j] - a[j] * u[j - 1]) / bet;
    }
    for (j = n - 2; j >= 0; j--) {
        u[j] -= gam[j + 1] * u[j + 1];
    }

    free(gam);
}
