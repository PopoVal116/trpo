#include "gemm.h"
#include <complex.h>
#include <stdio.h>

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
) {
    if (Order != CblasRowMajor || TransA != CblasNoTrans || TransB != CblasNoTrans) {
        printf("sgemm: Only RowMajor + NoTrans supported\n");
        return;
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*lda + k] * B[k*ldb + j];
            }
            C[i*ldc + j] = alpha * sum + beta * C[i*ldc + j];
        }
    }
}

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
) {
    if (Order != CblasRowMajor || TransA != CblasNoTrans || TransB != CblasNoTrans) {
        printf("dgemm: Only RowMajor + NoTrans supported\n");
        return;
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i*lda + k] * B[k*ldb + j];
            }
            C[i*ldc + j] = alpha * sum + beta * C[i*ldc + j];
        }
    }
}

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
) {
    if (Order != CblasRowMajor || TransA != CblasNoTrans || TransB != CblasNoTrans) {
        printf("cgemm: Only RowMajor + NoTrans supported\n");
        return;
    }
    const float complex *alpha_c = (const float complex *)alpha;
    const float complex *beta_c  = (const float complex *)beta;
    const float complex *A_c = (const float complex *)A;
    const float complex *B_c = (const float complex *)B;
    float complex *C_c = (float complex *)C;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float complex sum = 0.0f + 0.0f * I;
            for (int k = 0; k < K; k++) {
                sum += A_c[i*lda + k] * B_c[k*ldb + j];
            }
            C_c[i*ldc + j] = (*alpha_c) * sum + (*beta_c) * C_c[i*ldc + j];
        }
    }
}

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
) {
    if (Order != CblasRowMajor || TransA != CblasNoTrans || TransB != CblasNoTrans) {
        printf("zgemm: Only RowMajor + NoTrans supported\n");
        return;
    }
    const double complex *alpha_c = (const double complex *)alpha;
    const double complex *beta_c  = (const double complex *)beta;
    const double complex *A_c = (const double complex *)A;
    const double complex *B_c = (const double complex *)B;
    double complex *C_c = (double complex *)C;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double complex sum = 0.0 + 0.0 * I;
            for (int k = 0; k < K; k++) {
                sum += A_c[i*lda + k] * B_c[k*ldb + j];
            }
            C_c[i*ldc + j] = (*alpha_c) * sum + (*beta_c) * C_c[i*ldc + j];
        }
    }
}