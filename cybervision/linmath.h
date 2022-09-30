#ifndef LINMATH_H
#define LINMATH_H

typedef void* svd_internal;
svd_internal init_svd();
void free_svd(svd_internal);
int svdf(svd_internal ctx, float *matrix, int rows, int cols, float *u, float *s, float *v);
void multiplyf(float *a, float *b, float *output, int m, int n, int k, int transposeA, int transposeB);

#endif
