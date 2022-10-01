#include <stdlib.h>
#include <f2c.h>
#include <clapack.h>

#include "linmath.h"

typedef struct {
    integer work_size, iwork_size;
    float *work;
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
int svdf(svd_internal svd, float *matrix, int rows, int cols, float *u, float *s, float *v)
{
    // TODO: reuse/preallocate work memory
    integer info;
    float optimal_work;
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
    int result = sgesdd_("A", &m, &n, matrix, &lda, s, u, &ldu, v, &ldvt, &optimal_work, &lwork, ctx->iwork, &info);
    if (info != 0)
        return 0;
    lwork = (int)optimal_work;
    if (ctx->work_size < lwork)
    {
        size_t new_size = sizeof(float)*lwork;
        ctx->work = ctx->work == NULL? malloc(new_size) : realloc(ctx->work, new_size);
        ctx->work_size = lwork;
    }
    result = sgesdd_("A", &m, &n, matrix, &lda, s, u, &ldu, v, &ldvt, ctx->work, &lwork, ctx->iwork, &info);
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
