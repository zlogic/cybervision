#ifndef LINMATH_H
#define LINMATH_H

#include <stddef.h>

void multiply_matrix_3x3(float a[9], float b[9], float output[9]);
void multiply_matrix_3tx3(float at[9], float b[9], float output[9]);

void multiply_f_vector(float fundamental_matrix[9], float point[3], float target[3]);
void multiply_ft_vector(float fundamental_matrix[9], float point[3], float target[3]);

typedef void* svd_internal;
svd_internal init_svd();
void free_svd(svd_internal);
int svd(svd_internal ctx, double *matrix, const size_t rows, const size_t cols, double *u, double *s, double *vt);

#endif
