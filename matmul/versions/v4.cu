// 5291 Gflops
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

void rand_init(float *M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float random_float = (float)rand() / (float)RAND_MAX;
            M[n*i+j] = random_float;
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

const int BLOCKSIZE = 32;

__global__ void cuda_matmul(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * BLOCKSIZE + threadIdx.y;
    int j = blockIdx.y * BLOCKSIZE + threadIdx.x; // so consecutive j will be same warp
    // for some reason swapping blockIdx.x and blockIdx.y here results in a slight slowdown.

    __shared__ float As[32 * 32];
    __shared__ float Bs[32 * 32];

    float tmp = 0;
    for (int l = 0; l < 4096/32; l++) {

        As[32*threadIdx.y+threadIdx.x] = A[n*i+32*l+threadIdx.x];
        Bs[32*threadIdx.y+threadIdx.x] = B[n*(32*l+threadIdx.y)+j];

        __syncthreads();

        for (int k = 0; k < 32; k++) {
            tmp += As[32*threadIdx.y+k] * Bs[32*k+threadIdx.x];
        }
    }
    C[n*i+j] = tmp;
}


void perform_matmul(float *dA, float *dB, float *dC, int n) {

    cudaError_t err;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(n/BLOCKSIZE, n/BLOCKSIZE);
    cuda_matmul<<<gridDim, blockDim>>>(dA, dB, dC, n);
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

    //rand_init(hA, n);
    //rand_init(hB, n);

    float *dA = cuda_matrix(n);
    float *dB = cuda_matrix(n);
    float *dC = cuda_matrix(n);

    cpu_to_cuda(hA, dA, n);
    cpu_to_cuda(hB, dB, n);

    // -----

    perform_matmul(dA, dB, dC, n); // warmup? doesn't seem to actually reduce variance

    struct timespec start, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int steps = 100;
    for (int step = 0; step < steps; step++) {
        perform_matmul(dA, dB, dC, n);
    }
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec);
    elapsed += (end.tv_nsec - start.tv_nsec) / 1e9;
    float per_step = elapsed/steps;
    printf("Time elapsed: %.6f seconds per step\n", per_step);
    double flops = n*2/per_step;
    flops *= n*n; // split into two steps to avoid casting stuff
    double gflops = flops / 1000000000;
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

