#ifndef LINMATH_H
#define LINMATH_H

typedef void* svd_internal;
svd_internal init_svd();
void free_svd(svd_internal);
int svdd(svd_internal ctx, double *matrix, int rows, int cols, double *s, double *v);
void multiply_matrix_3x3(float a[9], float b[9], float output[9]);
void multiply_matrix_3tx3(float at[9], float b[9], float output[9]);

void multiply_f_vector(float fundamental_matrix[9], float point[3], float target[3]);
void multiply_ft_vector(float fundamental_matrix[9], float point[3], float target[3]);

#endif
