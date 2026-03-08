#include "gemm.h"
#include <complex.h>

void my_sgemm(
    const enum CBLAC_ORDER Order,
    const enum CBLAS_TRANSPONE TransA,
    const enum CBLAS_TRANSPONE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float *A,
    const int lda,
    const float *B,
    const int ldb,
    const float beta,
    float *C,
    const int ldc
) {
    if (Order != CblasRowMajor || TransA != CblasNoTrans || TransB != CblasNoTrans) {
        printf("sgemm: Only RowMajor + NoTrans supported\n");
        return;
    }
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            float sum = 0.0f;
            for (int k = 0; k < K; k++){
                sum += A[i*lda + k] * B[k*ldb + j];
        
            }
            C[i*ldc + j] = alpha*sum + beta*C[i*ldc + j];
        }
    }
}

void my_dgemm(
    const enum CBLAC_ORDER Order,
    const enum CBLAS_TRANSPONE TransA,
    const enum CBLAS_TRANSPONE TransB,
    const int M,
    const int N,
    const int K,
    const double alpha,
    const double *A,
    const int lda,
    const double *B,
    const int ldb,
    const double beta,
    double *C,
    const int ldc
) {
    if (Order != CblasRowMajor || TransA != CblasNoTrans || TransB != CblasNoTrans) {
        printf("dgemm: Only RowMajor + NoTrans supported\n");
        return;
    }
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            double sum = 0.0;
            for (int k = 0; k < K; k++){
                sum += A[i*lda + k] * B[k*ldb + j];
        
            }
            C[i*ldc + j] = alpha*sum + beta*C[i*ldc + j];
        }
    }
}

void my_cgemm(
    const enum CBLAC_ORDER Order,
    const enum CBLAS_TRANSPONE TransA,
    const enum CBLAS_TRANSPONE TransB,
    const int M,
    const int N,
    const int K,
    const void *alpha,
    const void *A,
    const int lda,
    const void *B,
    const int ldb,
    const void *beta,
    void *C,
    const int ldc
) {
    if (Order != CblasRowMajor || TransA != CblasNoTrans || TransB != CblasNoTrans) {
        printf("cgemm: Only RowMajor + NoTrans supported\n");
        return;
    }
    const float complex *alpha_c = (const float complex *)alpha;
    const float complex *beta_c = (const float complex *)beta;
    const float complex *A_c = (const float complex *)A;
    const float complex *B_c = (const float complex *)B;
     float complex *C_c = (float complex *)C;

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            float complex sum = 0.0f + 0.0f * I;
            for (int k = 0; k < K; k++){
                sum += A_c[i*lda + k] * B_c[k*ldb + j];
        
            }
            C_c[i*ldc + j] = (*alpha_c)*sum + (*beta_c)*C_c[i*ldc + j];
        }
    }
}

void my_zgemm(
    const enum CBLAC_ORDER Order,
    const enum CBLAS_TRANSPONE TransA,
    const enum CBLAS_TRANSPONE TransB,
    const int M,
    const int N,
    const int K,
    const void *alpha,
    const void *A,
    const int lda,
    const void *B,
    const int ldb,
    const void *beta,
    void *C,
    const int ldc
) {
    if (Order != CblasRowMajor || TransA != CblasNoTrans || TransB != CblasNoTrans) {
        printf("zgemm: Only RowMajor + NoTrans supported\n");
        return;
    }
    const double complex *alpha_c = (const double complex *)alpha;
    const double complex *beta_c = (const double complex *)beta;
    const double complex *A_c = (const double complex *)A;
    const double complex *B_c = (const double complex *)B;
    double complex *C_c = (double complex *)C;

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            double complex sum = 0.0 + 0.0 * I;
            for (int k = 0; k < K; k++){
                sum += A_c[i*lda + k] * B_c[k*ldb + j];
        
            }
            C_c[i*ldc + j] = (*alpha_c)*sum + (*beta_c)*C_c[i*ldc + j];
        }
    }
}