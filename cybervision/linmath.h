#ifndef LINMATH_H
#define LINMATH_H

typedef void* svd_internal;
svd_internal init_svd();
void free_svd(svd_internal);
int svdd(svd_internal ctx, double *matrix, int rows, int cols, double *u, double *s, double *v);
void multiplyd(double *a, double *b, double *output, int m, int n, int k, int transposeA, int transposeB);

#endif
