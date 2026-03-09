#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cblas.h>
#include "../include/gemm.h"
#include <complex.h>
#define NUM_RUNS 10
#define MIN_TIME_SEC 60.0

void perf_dgemm(){
    int m = 1024;
    int n = 1024;
    int k = 1024;

    double *A = malloc(m * k * sizeof(double));
    double *B = malloc(k * n * sizeof(double));
    double *C_open = malloc(m * n * sizeof(double));
    double *C_my = malloc(m * n * sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < m * k; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (double)rand() / RAND_MAX;
    }

    int threads[] = {1, 2, 4, 8, 16};
    printf("Тест производительности dgemm\n");
    printf("Размер матриц: %d x %d x %d\n", m, n, k);
    printf("Количество запусков: %d\n\n", NUM_RUNS);

    for (int t = 0; t < 5; t++) {
        omp_set_num_threads(threads[t]);
        openblas_set_num_threads(threads[t]);

        double time_open_total = 0.0;
        double time_my_total = 0.0;

        for (int run =0; run <NUM_RUNS; run++){
            clock_t start = clock();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C_open, n);
            time_open_total += (double)(clock() - start) / CLOCKS_PER_SEC;

            start = clock();
            my_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C_my, n);
            time_my_total += (double)(clock() - start) / CLOCKS_PER_SEC;
        }
        double avg_time_open = time_open_total / NUM_RUNS;
        double avg_time_my = time_my_total / NUM_RUNS;

        double gflops_open = 2.0 * m * n * k / avg_time_open / 1e9;
        double gflops_my = 2.0 * m * n * k / avg_time_my / 1e9;
        double percent = (gflops_my / gflops_open) * 100;

        printf("Потоки: %d\n", threads[t]);
        printf("  OpenBLAS: %.2f GFLOPS (ср. время %.3f сек)\n", gflops_open, avg_time_open);
        printf("  Моя версия: %.2f GFLOPS (ср. время %.3f сек)\n", gflops_my, avg_time_my);
        printf("  Относительная производительность: %.1f%%\n\n", percent);
    }

    free(A);
    free(B);
    free(C_open);
    free(C_my);
}

void perf_sgemm(){
int m = 1024;
    int n = 1024;
    int k = 1024;

    float *A = malloc(m * k * sizeof(float));
    float *B = malloc(k * n * sizeof(float));
    float *C_open = malloc(m * n * sizeof(float));
    float *C_my = malloc(m * n * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < m * k; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }

    int threads[] = {1, 2, 4, 8, 16};
    printf("Тест производительности sgemm\n");
    printf("Размер матриц: %d x %d x %d\n", m, n, k);
    printf("Количество запусков: %d\n\n", NUM_RUNS);

    for (int t = 0; t < 5; t++) {
        omp_set_num_threads(threads[t]);
        openblas_set_num_threads(threads[t]);

        double time_open_total = 0.0;
        double time_my_total = 0.0;

        for (int run =0; run <NUM_RUNS; run++){
            clock_t start = clock();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A, k, B, n, 0.0, C_open, n);
            time_open_total += (double)(clock() - start) / CLOCKS_PER_SEC;

            start = clock();
            my_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A, k, B, n, 0.0, C_my, n);
            time_my_total += (double)(clock() - start) / CLOCKS_PER_SEC;
        }
        double avg_time_open = time_open_total / NUM_RUNS;
        double avg_time_my = time_my_total / NUM_RUNS;

        double gflops_open = 2.0 * m * n * k / avg_time_open / 1e9;
        double gflops_my = 2.0 * m * n * k / avg_time_my / 1e9;
        double percent = (gflops_my / gflops_open) * 100;

        printf("Потоки: %d\n", threads[t]);
        printf("  OpenBLAS: %.2f GFLOPS (ср. время %.3f сек)\n", gflops_open, avg_time_open);
        printf("  Моя версия: %.2f GFLOPS (ср. время %.3f сек)\n", gflops_my, avg_time_my);
        printf("  Относительная производительность: %.1f%%\n\n", percent);
    }

    free(A);
    free(B);
    free(C_open);
    free(C_my);
}

void perf_cgemm(){
   int m = 1024;
    int n = 1024;
    int k = 1024;

    float complex *A = malloc(m * k * sizeof(float complex));
    float complex *B = malloc(k * n * sizeof(float complex));
    float complex *C_open = malloc(m * n * sizeof(float complex));
    float complex *C_my = malloc(m * n * sizeof(float complex));

    float complex alpha = 1.0f + 0.0f * I;
    float complex beta = 0.0f + 0.0f * I;

    srand(time(NULL));
    for (int i = 0; i < m * k; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }

    int threads[] = {1, 2, 4, 8, 16};
    printf("Тест производительности cgemm\n");
    printf("Размер матриц: %d x %d x %d\n", m, n, k);
    printf("Количество запусков: %d\n\n", NUM_RUNS);

    for (int t = 0; t < 5; t++) {
        omp_set_num_threads(threads[t]);
        openblas_set_num_threads(threads[t]);

        double time_open_total = 0.0;
        double time_my_total = 0.0;

        for (int run =0; run <NUM_RUNS; run++){
            clock_t start = clock();
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, A, k, B, n, &beta, C_open, n);
            time_open_total += (double)(clock() - start) / CLOCKS_PER_SEC;

            start = clock();
            my_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, A, k, B, n, &beta, C_my, n);
            time_my_total += (double)(clock() - start) / CLOCKS_PER_SEC;
        }
        double avg_time_open = time_open_total / NUM_RUNS;
        double avg_time_my = time_my_total / NUM_RUNS;

        double gflops_open = 2.0 * m * n * k / avg_time_open / 1e9;
        double gflops_my = 2.0 * m * n * k / avg_time_my / 1e9;
        double percent = (gflops_my / gflops_open) * 100;

        printf("Потоки: %d\n", threads[t]);
        printf("  OpenBLAS: %.2f GFLOPS (ср. время %.3f сек)\n", gflops_open, avg_time_open);
        printf("  Моя версия: %.2f GFLOPS (ср. время %.3f сек)\n", gflops_my, avg_time_my);
        printf("  Относительная производительность: %.1f%%\n\n", percent);
    }

    free(A);
    free(B);
    free(C_open);
    free(C_my);
}

void perf_zgemm(){
    int m = 1024;
    int n = 1024;
    int k = 1024;

    double complex *A = malloc(m * k * sizeof(double complex));
    double complex *B = malloc(k * n * sizeof(double complex));
    double complex *C_open = malloc(m * n * sizeof(double complex));
    double complex *C_my = malloc(m * n * sizeof(double complex));

    double complex alpha = 1.0 + 0.0 * I;
    double complex beta = 0.0 + 0.0 * I;

    srand(time(NULL));
    for (int i = 0; i < m * k; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (double)rand() / RAND_MAX;
    }

    int threads[] = {1, 2, 4, 8, 16};
    printf("Тест производительности zgemm\n");
    printf("Размер матриц: %d x %d x %d\n", m, n, k);
    printf("Количество запусков: %d\n\n", NUM_RUNS);

    for (int t = 0; t < 5; t++) {
        omp_set_num_threads(threads[t]);
        openblas_set_num_threads(threads[t]);

        double time_open_total = 0.0;
        double time_my_total = 0.0;

        for (int run =0; run <NUM_RUNS; run++){
            clock_t start = clock();
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, A, k, B, n, &beta, C_open, n);
            time_open_total += (double)(clock() - start) / CLOCKS_PER_SEC;

            start = clock();
            my_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, A, k, B, n, &beta, C_my, n);
            time_my_total += (double)(clock() - start) / CLOCKS_PER_SEC;
        }
        double avg_time_open = time_open_total / NUM_RUNS;
        double avg_time_my = time_my_total / NUM_RUNS;

        double gflops_open = 2.0 * m * n * k / avg_time_open / 1e9;
        double gflops_my = 2.0 * m * n * k / avg_time_my / 1e9;
        double percent = (gflops_my / gflops_open) * 100;

        printf("Потоки: %d\n", threads[t]);
        printf("  OpenBLAS: %.2f GFLOPS (ср. время %.3f сек)\n", gflops_open, avg_time_open);
        printf("  Моя версия: %.2f GFLOPS (ср. время %.3f сек)\n", gflops_my, avg_time_my);
        printf("  Относительная производительность: %.1f%%\n\n", percent);
    }

    free(A);
    free(B);
    free(C_open);
    free(C_my);
}

int main() {
    printf("sgemm\n");
    perf_sgemm();
    printf("\ndgemm\n");
    perf_dgemm();
    printf("\ncgemm\n");
    perf_cgemm();
    printf("\nzgemm\n");
    perf_zgemm();
    return 0;
}