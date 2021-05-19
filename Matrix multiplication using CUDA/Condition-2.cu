#include <cublas_v2.h>
#include <curand.h>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include "addvec.h"

// Here i am checking the code for cpu also.
// CUBLAS operating on column-major data
void verify_solution(float* a, float* b, float* c, int M, int N, int K) {
    float epsilon = 0.1f;
    // each row of the matrix
    for (int row = 0; row < M; row++) {
        // each column of the matrix
        for (int col = 0; col < N; col++) {
            // each row and column calculation linewise.
            float temp = 0;
            for (int i = 0; i < K; i++) {
                temp += a[row + M * i] * b[col * K + i];
            }
            // difference check
            assert(fabs(c[col * M + row] - temp) <= epsilon);
        }
    }
}

int matmul(const int M, const int N, const int K, const int iter) {

    // random number generator using Pseudo 
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // seed setting
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    
    // for static B, i am generating random values for B only one time.
    float* d_b;
    const size_t bytes_b = K * N * sizeof(float);
    cudaMalloc(&d_b, bytes_b);
    // Filling matrix B with random numbers 
    curandGenerateUniform(prng, d_b, K * M);
    
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
        const size_t bytes_c = M * N * sizeof(float);

        // Vectors for the host data
        std::vector<float> h_a(M * K);
        std::vector<float> h_b(K * N);
        std::vector<float> h_c(M * N);

        // Allocate device memory
        float* d_a, * d_c;
        cudaMalloc(&d_a, bytes_a);
        cudaMalloc(&d_b, bytes_b);
        cudaMalloc(&d_c, bytes_c);

        // Filling matrix A with random numbers 
        curandGenerateUniform(prng, d_a, M * K);

        // cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Scalaing factors
        float alpha = 1.0f;
        float beta = 0.0f;

        // matrix calculation using cublas sgemm.
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, M, d_b, K, &beta, d_c, M);

        // Copy back the three matrices
        cudaMemcpy(h_a.data(), d_a, bytes_a, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b.data(), d_b, bytes_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

        // Verify solution
        verify_solution(h_a.data(), h_b.data(), h_c.data(), M, N, K);
        //std::cout << "COMPLETED SUCCESSFULLY\n";

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
    printf("The elapsed time in gpu was %.2f ms\ ", elapsed);
    return 0;
}
int main() {

    const int M3 = 50;
    const int N3 = 50;
    const int K3 = 20;
    const int iter3 = 5000;

    matmul(M3, N3, K3, iter3);
    std::cout << "Second condition COMPLETED SUCCESSFULLY\n";
    return 0;
}