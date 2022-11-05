#ifndef LINMATH_H
#define LINMATH_H

#include <stddef.h>

void multiply_matrix_3x3(double a[9], double b[9], double output[9]);
void multiply_matrix_3tx3(double at[9], double b[9], double output[9]);

void multiply_f_vector(double fundamental_matrix[9], double point[3], double target[3]);
void multiply_ft_vector(double fundamental_matrix[9], double point[3], double target[3]);
void multiply_p_vector(double projection_matrix[12], double point[4], double target[3]);

void multiplyd(double *a, double *b, double *output, int m, int n, int k, int transposeA, int transposeB);

typedef void* svd_internal;
svd_internal init_svd();
void free_svd(svd_internal);
int svdd(svd_internal ctx, double *matrix, int rows, int cols, double *u, double *s, double *v);

typedef void* invert_internal;
invert_internal init_invert();
void free_invert(invert_internal);
int invertd(invert_internal ctx, double *a, int n);

#endif
