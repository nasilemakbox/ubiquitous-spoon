#include "check.h"
#include "heptdag.h"


void heptdag(double a[], double b[], double c[], double d[], double e[],
             double f[], double g[], double h[], double x[], int n)
{
    int i;
    double bet, alp, den;
    double *p = (double *) malloc(sizeof(double) * n);
    double *q = (double *) malloc(sizeof(double) * n);
    double *r = (double *) malloc(sizeof(double) * n);
    double *s = x; // Alias

    check(p != NULL);
    check(q != NULL);
    check(r != NULL);

    den = d[0];
    check(den != 0.0);
    p[0] = e[0] / den;
    q[0] = f[0] / den;
    r[0] = g[0] / den;
    s[0] = h[0] / den;

    den = d[1] - c[1] * p[0];
    check(den != 0.0);
    p[1] = (e[1] - c[1] * q[0]) / den;
    q[1] = (f[1] - c[1] * r[0]) / den;
    r[1] = (g[1]              ) / den;
    s[1] = (h[1] - c[1] * s[0]) / den;

    bet = c[2] - b[2] * p[0];
    den = d[2] - b[2] * q[0] - bet * p[1];
    check(den != 0.0);
    p[2] = (e[2] - b[2] *r[0] - bet *q[1]) / den;
    q[2] = (f[2]              - bet *r[1]) / den;
    r[2] = (g[2]                         ) / den;
    s[2] = (h[2] - b[2] *s[0] - bet *s[1]) / den;

    for (i = 3; i < n; i++) {
        alp = b[i] - a[i] * p[i - 3];
        bet = c[i] - a[i] * q[i - 3] - alp * p[i - 2];
        den = d[i] - a[i] * r[i - 3] - alp * q[i - 2] - bet * p[i - 1];
        check(den != 0.0);
        p[i] = (e[i]                   - alp * r[i - 2] - bet * q[i - 1]) / den;
        q[i] = (f[i]                                    - bet * r[i - 1]) / den;
        r[i] = (g[i]                                                    ) / den;
        s[i] = (h[i] - a[i] * s[i - 3] - alp * s[i - 2] - bet * s[i - 1]) / den;
    }
    x[n - 1] = s[n - 1];
    x[n - 2] = s[n - 2] - p[n - 2] * s[n - 1];
    x[n - 3] = s[n - 3] - p[n - 3] * s[n - 2] - q[n - 3] * s[n - 1];
    for (i = n - 4; i >= 0; i--) {
        x[i] = s[i] - p[i] * s[i + 1] - q[i] * s[i + 2] - r[i] * s[i + 3];
    }

    free(p);
    free(q);
    free(r);
}
