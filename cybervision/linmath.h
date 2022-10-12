#ifndef LINMATH_H
#define LINMATH_H

typedef void* svd_internal;
svd_internal init_svd();
void free_svd(svd_internal);
int svdd(svd_internal ctx, double *matrix, int rows, int cols, double *s, double *v);
void multiply_matrix_3x3(double a[9], double b[9], double output[9]);
void multiply_matrix_3tx3(double at[9], double b[9], double output[9]);

void multiply_f_vector(double fundamental_matrix[9], double point[3], double target[3]);
void multiply_ft_vector(double fundamental_matrix[9], double point[3], double target[3]);

#endif
