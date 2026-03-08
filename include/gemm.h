#ifndef GEMM_H
#define GEMM_H

#include <cblas.h>

void my_sgemm(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const float alpha,
    const float *A, const int lda,
    const float *B, const int ldb,
    const float beta,
    float *C, const int ldc
);

void my_dgemm(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha,
    const double *A, const int lda,
    const double *B, const int ldb,
    const double beta,
    double *C, const int ldc
);

void my_cgemm(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const void *beta,
    void *C, const int ldc
);

void my_zgemm(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const void *alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const void *beta,
    void *C, const int ldc
);

#endif