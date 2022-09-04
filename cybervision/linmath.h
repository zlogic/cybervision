#ifndef LINMATH_H
#define LINMATH_H

int svdf(float *matrix, int rows, int cols, float *u, float *s, float *v);
void multiplyf(float *a, float *b, float *output, int m, int n, int k, int transposeA, int transposeB);

#endif
