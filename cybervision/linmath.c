#include <stdlib.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
typedef __CLPK_integer integer;
#else
#include <f2c.h>
#include <clapack.h>
#endif

#include "linmath.h"

void multiply_matrix_3x3(double a[9], double b[9], double output[9])
{
    for(size_t i=0;i<3;i++)
        for(size_t j=0;j<3;j++)
            output[i*3+j] = a[i*3+0]*b[0*3+j] + a[i*3+1]*b[1*3+j] + a[i*3+2]*b[2*3+j];
}

void multiply_matrix_3tx3(double at[9], double b[9], double output[9])
{
    for(size_t i=0;i<3;i++)
        for(size_t j=0;j<3;j++)
            output[i*3+j] = at[0*3+i]*b[0*3+j] + at[1*3+i]*b[1*3+j] + at[2*3+i]*b[2*3+j];
}

void multiply_f_vector(double f[9], double p[3], double target[3])
{
    target[0] = f[0]*p[0]+f[1]*p[1]+f[2]*p[2];
    target[1] = f[3]*p[0]+f[4]*p[1]+f[5]*p[2];
    target[2] = f[6]*p[0]+f[7]*p[1]+f[8]*p[2];
}

void multiply_ft_vector(double f[9], double p[3], double target[3])
{
    target[0] = f[0]*p[0]+f[3]*p[1]+f[6]*p[2];
    target[1] = f[1]*p[0]+f[4]*p[1]+f[7]*p[2];
    target[2] = f[2]*p[0]+f[5]*p[1]+f[8]*p[2];
}

void multiply_p_vector(double p[12], double v[4], double target[3])
{
    target[0] = p[0]*v[0]+p[1]*v[1]+ p[2]*v[2]+ p[3]*v[3];
    target[1] = p[4]*v[0]+p[5]*v[1]+ p[6]*v[2]+ p[7]*v[3];
    target[2] = p[8]*v[0]+p[9]*v[1]+p[10]*v[2]+p[11]*v[3];
}

typedef struct {
    integer work_size, iwork_size;
    double *work;
    integer *iwork;
} svd_ctx;

svd_internal init_svd()
{
    svd_ctx *ctx = malloc(sizeof(svd_ctx));
    ctx->work = NULL;
    ctx->iwork = NULL;
    ctx->work_size = 0;
    ctx->iwork_size = 0;
    return ctx;
}
void free_svd(svd_internal svd)
{
    svd_ctx *ctx = svd;
    if (ctx == NULL)
        return;
    if (ctx->work != NULL)
        free(ctx->work);
    if (ctx->iwork != NULL)
        free(ctx->iwork);
    free(ctx);
}
int svdd(svd_internal svd, double *matrix, int rows, int cols, double *u, double *s, double *v)
{
    // Warning: LAPACK needs column-major matrices (iterate by columns, first, then by rows)
    integer info = 0;
    double optimal_work = 0.0;
    integer m = rows, n = cols;
    integer lda = m, ldu = m, ldvt = n;
    integer lwork = -1;
    svd_ctx *ctx = svd;
    size_t iwork_size = 8*(m<n? m:n);
    if (ctx->iwork_size < iwork_size)
    {
        size_t new_size = sizeof(integer)*iwork_size;
        ctx->iwork = ctx->iwork == NULL? malloc(new_size) : realloc(ctx->iwork, new_size);
        ctx->iwork_size = iwork_size;
    }
    int result = dgesdd_("A", &m, &n, matrix, &lda, s, u, &ldu, v, &ldvt, &optimal_work, &lwork, ctx->iwork, &info);
    if (info != 0)
        return 0;
    lwork = (int)optimal_work;
    if (ctx->work_size < lwork)
    {
        size_t new_size = sizeof(double)*lwork;
        ctx->work = ctx->work == NULL? malloc(new_size) : realloc(ctx->work, new_size);
        ctx->work_size = lwork;
    }
    result = dgesdd_("A", &m, &n, matrix, &lda, s, u, &ldu, v, &ldvt, ctx->work, &lwork, ctx->iwork, &info);
    return info == 0;
}

void multiplyd(double *a, double *b, double *output, int m, int n, int k, int transposeA, int transposeB)
{
    // Warning: LAPACK needs column-major matrices (iterate by columns, first, then by rows)
    integer m_in = m, n_in = n, k_in = k;
    double alpha = 1.0;
    double beta = 0.0;
    integer lda = transposeA? k:m;
    integer ldb = transposeB? n:k;
    integer ldc = m;
    dgemm_(transposeA? "T":"N", transposeB? "T":"N", &m_in, &n_in, &k_in, &alpha, a, &lda, b, &ldb, &beta, output, &ldc);
}

typedef struct {
    integer ipiv_size;
    integer lwork_size;
    integer *ipiv;
    double *lwork;
} invert_ctx;
invert_internal init_invert()
{
    invert_ctx *ctx = malloc(sizeof(invert_ctx));
    ctx->ipiv = NULL;
    ctx->lwork = NULL;
    ctx->ipiv_size = 0;
    ctx->lwork_size = 0;
    return ctx;
}
void free_invert(invert_internal invert)
{
    invert_ctx *ctx = invert;
    if (ctx == NULL)
        return;
    if (ctx->ipiv != NULL)
        free(ctx->ipiv);
    if (ctx->lwork != NULL)
        free(ctx->lwork);
    free(ctx);
}
int invertd(invert_internal invert, double *a, int n)
{
    // Warning: LAPACK needs column-major matrices (iterate by columns, first, then by rows)
    int result = 0;
    integer info = 0;
    integer n_in = n;
    integer lda = n;
    integer lwork = -1;
    double optimal_work = 0.0;
    invert_ctx *ctx = invert;
    if (ctx->ipiv_size < n)
    {
        ctx->ipiv_size = n;
        size_t ipiv_size = sizeof(integer)*ctx->ipiv_size;
        ctx->ipiv = ctx->ipiv == NULL? malloc(ipiv_size) : realloc(ctx->ipiv, ipiv_size);
    }
    dgetrf_(&n_in, &n_in, a, &lda, ctx->ipiv, &info);
    if (info != 0)
        return 0;

    dgetri_(&n_in, a, &lda, ctx->ipiv, &optimal_work, &lwork, &info);
    if (info != 0)
        return 0;
    lwork = (int)optimal_work;

    if (ctx->lwork_size < lwork)
    {
        ctx->lwork_size = lwork;
        size_t lwork_size = sizeof(double)*lwork;
        ctx->lwork = ctx->lwork == NULL? malloc(lwork_size) : realloc(ctx->lwork, lwork_size);
    }
    dgetri_(&n_in, a, &lda, ctx->ipiv, ctx->lwork, &lwork, &info);
    return info == 0;
}
