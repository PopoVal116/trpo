#ifndef GEMM_H
#define GEMM_H

#include <cblas.h>

void my_sgemm(
    enum CBLAS_ORDER Order,
    enum CBLAS_TRANSPOSE TransA,  
    enum CBLAS_TRANSPOSE TransB,
    int M, int N, int K,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc
);

void my_dgemm(
    enum CBLAS_ORDER Order,
    enum CBLAS_TRANSPOSE TransA,
    enum CBLAS_TRANSPOSE TransB,
    int M, int N, int K,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double beta,
    double *C, int ldc
);

void my_cgemm(
    enum CBLAS_ORDER Order,
    enum CBLAS_TRANSPOSE TransA,
    enum CBLAS_TRANSPOSE TransB,
    int M, int N, int K,
    const void *alpha,
    const void *A, int lda,
    const void *B, int ldb,
    const void *beta,
    void *C, int ldc
);

void my_zgemm(
    enum CBLAS_ORDER Order,
    enum CBLAS_TRANSPOSE TransA,
    enum CBLAS_TRANSPOSE TransB,
    int M, int N, int K,
    const void *alpha,
    const void *A, int lda,
    const void *B, int ldb,
    const void *beta,
    void *C, int ldc
);

#endif