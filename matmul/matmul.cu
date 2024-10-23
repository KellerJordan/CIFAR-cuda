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

void zero_init(float *M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[n*i+j] = 0;
        }
    }
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
    //int i = blockIdx.x * blockDim.x + threadIdx.x / blockDim.x;
    //int j = blockIdx.y * blockDim.y + threadIdx.x / blockDim.y;
    //int idx = n*i + j;
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
    //dim3 gridDim(n/32, n/32, 1);
    //cuda_matmul<<<gridDim, threadsPerBlock>>>(dA, dB, dC, n);
    int blocksPerGrid = n*n/threadsPerBlock;
    cuda_matmul<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*
    float *hC = (float *)malloc(n*n*sizeof(float));
    err = cudaMemcpy(hC, dC, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "line 56 Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("%f\n", hC[0]);
    printf("%f\n", hC[n]);
    */
}

void check_result(float *dC, int n) {
    size_t size = n*n*sizeof(float);
    float *hC = (float *)malloc(size);
    cudaError_t err = cudaMemcpy(hC, dC, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "line 56 Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float total = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            total += hC[n*i+j];
        }
    }
    printf("Total: %f (should be 1252)\n", total);
    free(hC);
}

int main() {

    int n = 4096;

    float *hA = cpu_matrix(n);
    float *hB = cpu_matrix(n);
    zero_init(hA, n);
    zero_init(hB, n);
    hA[0] = 1;
    hA[1] = 3;
    hB[0] = 6;
    hB[1] = 5; // shouldn't matter
    hB[n] = 7;

    hA[777*n+999] = 20;
    hB[999*n+777] = 61;

    float *dA = cuda_matrix(n);
    float *dB = cuda_matrix(n);
    float *dC = cuda_matrix(n);

    cpu_to_cuda(hA, dA, n);
    cpu_to_cuda(hB, dB, n);

    // -----

    struct timespec start, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int steps = 5;
    for (int step = 0; step < steps; step++) {
        perform_matmul(dA, dB, dC, n);
    }
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec);
    elapsed += (end.tv_nsec - start.tv_nsec) / 1e9;
    float per_step = elapsed/steps;
    printf("Time elapsed: %.6f seconds per step\n", per_step);
    float flops = 2*n*n/per_step;
    float gflops = flops / 1000000000;
    printf("Gflops: %.3f\n", gflops);

    check_result(dC, n);

    // -----

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);

    cudaDeviceReset();
    return 0;
}

