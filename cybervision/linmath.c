#include <stdlib.h>
#include <f2c.h>
#include <clapack.h>

//#define sgemm_ sgemm

#include "linmath.h"

int svdf(float *matrix, int rows, int cols, float *u, float *s, float *v)
{
    // TODO: reuse/preallocate work memory
    integer info;
    float optimal_work;
    integer m = rows, n = cols;
    integer lda = m, ldu = u!=NULL?m:1, ldvt = n;
    integer lwork = -1;

    int result = sgesvd_(u!=NULL?"A":"N", "A", &m, &n, matrix, &lda, s, u, &ldu, v, &ldvt, &optimal_work, &lwork, &info);
    if (info != 0)
        return 0;
    lwork = (int)optimal_work;
    float *work = malloc(sizeof(float)*lwork);
    result = sgesvd_(u!=NULL?"A":"N", "A", &m, &n, matrix, &lda, s, u, &ldu, v, &ldvt, work, &lwork, &info);
    free(work);
    return info == 0;
}

void multiplyf(float *a, float *b, float *output, int m, int n, int k, int transposeA, int transposeB)
{
    integer m_in = m, n_in = n, k_in = k;
    float alpha = 1.0F;
    float beta = 0.0F;
    integer lda = transposeA? k:m;
    integer ldb = transposeB? n:k;
    integer ldc = m;
    sgemm_(transposeA? "T":"N", transposeB? "T":"N", &m_in, &n_in, &k_in, &alpha, a, &lda, b, &ldb, &beta, output, &ldc);
}
