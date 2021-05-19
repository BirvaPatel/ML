#include <cublas_v2.h>
#include <curand.h>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include "addvec.h"

int matmul(const int M, const int N, const int K, const int iter) {

    // random number generator using Pseudo 
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // seed setting
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    // calculation of gpu runtime using cuda event creator.
    float elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // loop to generate random A and C for given number of iteration.
    for (int n = 1; n <= iter; n++) {

        // Pre-calculate the size of our matrices
        const size_t bytes_a = M * K * sizeof(float);
        const size_t bytes_b = K * N * sizeof(float);
        const size_t bytes_c = M * N * sizeof(float);

        // Allocate device memory
        float* d_a, * d_c;
        float* d_b;
        cudaMalloc(&d_a, bytes_a);
        cudaMalloc(&d_b, bytes_b);
        cudaMalloc(&d_c, bytes_c);

        // Filling matrix A with random numbers 
        curandGenerateUniform(prng, d_a, M * K);
        // Filling matrix B with random numbers 
        curandGenerateUniform(prng, d_b, K * M);

        // cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Scalaing factors
        float alpha = 1.0f;
        float beta = 0.0f;

        // matrix calculation using cublas sgemm.
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, M, d_b, K, &beta, d_c, M);

        // Free our memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("time in gpu : %.2f ms\ ", elapsed);
    return 0;
}
int main() {
    // Condition -1 but with changing in matrix size and number of iterations.
    const int M = 5000;
    const int N = 5000;
    const int K = 4000;
    const int iter = 200;

    matmul(M, N, K, iter);
    std::cout << "Condition COMPLETED SUCCESSFULLY\n";
    return 0;
}