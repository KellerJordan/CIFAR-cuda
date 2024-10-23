#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<stdbool.h>

float *cpu_matrix(int n) {
    size_t size = n * n * sizeof(float);
    float *M = (float *)malloc(size);
    return M;
}

float *cuda_matrix(int n) {
    size_t size = n * n * sizeof(float);
    float *dM;
    cudaError_t err = cudaMalloc((void**)&dM, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return dM;
}

void cpu_to_cuda(float *hM, float *dM, int n) {
    size_t size = n*n*sizeof(float);
    cudaError_t err = cudaMemcpy(dM, hM, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "line 23 Failed to copy vector C from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void cuda_matmul(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / n;
    int j = idx % n;
    if (idx < n*n) {
        C[idx] = 0;
        for (int k = 0; k < n; k++) {
            C[idx] += A[n*i+k] * B[n*k+j];
        }
    }
}


void perform_matmul(float *dA, float *dB, float *dC, int n) {

    cudaError_t err;
    int threadsPerBlock = 1024;
    int blocksPerGrid = n*n/threadsPerBlock;
    cuda_matmul<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float hC;
    err = cudaMemcpy(&hC, dC, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "line 56 Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("%f\n", hC);
}

int main() {

    int n = 4096;

    float *hA = cpu_matrix(n);
    float *hB = cpu_matrix(n);
    hA[0] = 1;
    hA[1] = 3;
    hB[0] = 6;
    hB[1] = 5; // shouldn't matter
    hB[n] = 7;

    float *dA = cuda_matrix(n);
    float *dB = cuda_matrix(n);
    float *dC = cuda_matrix(n);

    cpu_to_cuda(hA, dA, n);
    cpu_to_cuda(hB, dB, n);

    // -----

    struct timespec start, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int steps = 20;
    for (int step = 0; step < steps; step++) {
        perform_matmul(dA, dB, dC, n);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec);
    elapsed += (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time elapsed: %.6f seconds per step\n", elapsed/steps);

    // -----

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);

    cudaDeviceReset();
    return 0;
}
