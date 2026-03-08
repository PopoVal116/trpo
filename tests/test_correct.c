#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "../include/gemm.h"

#define TOL_F 1e-5f
#define TOL_D 1e-10

void test_sgemm() {
    int m = 64, n = 64, k = 64;
    float *A = malloc(m * k * sizeof(float));
    float *B = malloc(k * n * sizeof(float));
    float *C_ref = calloc(m * n, sizeof(float));
    float *C_my = calloc(m * n, sizeof(float));

    srand(42);
    for (int i = 0; i < m * k; i++) {
        A[i] = (float)rand() / RAND_MAX * 2 - 1;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (float)rand() / RAND_MAX * 2 - 1;
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                A, k, B, n, 0.0f, C_ref, n);
    my_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
             A, k, B, n, 0.0f, C_my, n);

    int err = 0;
    for (int i = 0; i < m * n; i++) {
        if (fabsf(C_my[i] - C_ref[i]) > TOL_F) {
            err++;
        }
    }
    printf("sgemm errors: %d\n", err);

    free(A);
    free(B);
    free(C_ref);
    free(C_my);
}

void test_dgemm() {
    int m = 64, n = 64, k = 64;
    double *A = malloc(m * k * sizeof(double));
    double *B = malloc(k * n * sizeof(double));
    double *C_ref = calloc(m * n, sizeof(double));
    double *C_my = calloc(m * n, sizeof(double));

    srand(42);
    for (int i = 0; i < m * k; i++) {
        A[i] = (double)rand() / RAND_MAX * 2 - 1;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (double)rand() / RAND_MAX * 2 - 1;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                A, k, B, n, 0.0, C_ref, n);
    my_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
             A, k, B, n, 0.0, C_my, n);

    int err = 0;
    for (int i = 0; i < m * n; i++) {
        if (fabs(C_my[i] - C_ref[i]) > TOL_D) {
            err++;
        }
    }
    printf("dgemm errors: %d\n", err);

    free(A);
    free(B);
    free(C_ref);
    free(C_my);
}

void test_cgemm() {
    int m = 64, n = 64, k = 64;
    float complex *A = malloc(m * k * sizeof(float complex));
    float complex *B = malloc(k * n * sizeof(float complex));
    float complex *C_ref = calloc(m * n, sizeof(float complex));
    float complex *C_my  = calloc(m * n, sizeof(float complex));

    float complex alpha = 1.0f + 0.0f * I;
    float complex beta  = 0.0f + 0.0f * I;

    srand(42);
    for (int i = 0; i < m * k; i++) {
        A[i] = (float)rand() / RAND_MAX * 2 - 1 + I * ((float)rand() / RAND_MAX * 2 - 1);
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (float)rand() / RAND_MAX * 2 - 1 + I * ((float)rand() / RAND_MAX * 2 - 1);
    }

    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                A, k, B, n, &beta, C_ref, n);
    my_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
             A, k, B, n, &beta, C_my, n);

    int err = 0;
    for (int i = 0; i < m * n; i++) {
        if (cabsf(C_my[i] - C_ref[i]) > TOL_F) {
            err++;
        }
    }
    printf("cgemm errors: %d\n", err);

    free(A);
    free(B);
    free(C_ref);
    free(C_my);
}

void test_zgemm() {
    int m = 64, n = 64, k = 64;
    double complex *A = malloc(m * k * sizeof(double complex));
    double complex *B = malloc(k * n * sizeof(double complex));
    double complex *C_ref = calloc(m * n, sizeof(double complex));
    double complex *C_my  = calloc(m * n, sizeof(double complex));

    double complex alpha = 1.0 + 0.0 * I;
    double complex beta  = 0.0 + 0.0 * I;

    srand(42);
    for (int i = 0; i < m * k; i++) {
        A[i] = (double)rand() / RAND_MAX * 2 - 1 + I * ((double)rand() / RAND_MAX * 2 - 1);
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (double)rand() / RAND_MAX * 2 - 1 + I * ((double)rand() / RAND_MAX * 2 - 1);
    }

    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                A, k, B, n, &beta, C_ref, n);
    my_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
             A, k, B, n, &beta, C_my, n);

    int err = 0;
    for (int i = 0; i < m * n; i++) {
        if (cabs(C_my[i] - C_ref[i]) > TOL_D) {
            err++;
        }
    }
    printf("zgemm errors: %d\n", err);

    free(A);
    free(B);
    free(C_ref);
    free(C_my);
}

int main() {
    test_sgemm();
    test_dgemm();
    test_cgemm();
    test_zgemm();
    printf("Все тесты завершены\n");
    return 0;
}