#include <stdlib.h>
#include <f2c.h>
#include <clapack.h>

#include "linmath.h"

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
    integer info;
    double optimal_work;
    integer m = rows, n = cols;
    integer lda = m, ldu = m, ldvt = n;
    integer lwork = -1;
    svd_ctx *ctx = svd;
    integer iwork_size = 8*(m<n? m:n);
    if (ctx->iwork_size < iwork_size)
    {
        size_t new_size = sizeof(integer)*iwork_size;
        ctx->iwork = ctx->iwork == NULL? malloc(new_size) : realloc(ctx->work, new_size);
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
    integer m_in = m, n_in = n, k_in = k;
    double alpha = 1.0;
    double beta = 0.0;
    integer lda = transposeA? k:m;
    integer ldb = transposeB? n:k;
    integer ldc = m;
    dgemm_(transposeA? "T":"N", transposeB? "T":"N", &m_in, &n_in, &k_in, &alpha, a, &lda, b, &ldb, &beta, output, &ldc);
}
