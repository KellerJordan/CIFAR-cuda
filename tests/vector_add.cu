#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel
__global__ void vectorAdd(float *C, int N) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float s = 0;
        for (int i = 0; i < N; i++) {
            if (C[i] > 0)
                s += 1;
        }
        C[idx] = 1 + s / N;
    }
}

int main() {
    // Size of vectors
    int N = 1 << 20; // 1 Million elements

    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_C = (float*)malloc(size);

    // Allocate device memory
    float *d_C;
    cudaError_t err;

    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Start timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the vectorAdd CUDA Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_C, N);

    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for 1000 vector additions: %f ms\n", milliseconds);
    
    // Check for any errors launching the kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy device result vector C to host
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify the result
    printf("%f\n", h_C[1000]);
    float s = 0;
    for (int i = 0; i < N; i++) {
        s += h_C[i];
    }
    printf("%f\n", s / N);

    // Free device memory
    cudaFree(d_C);

    // Free host memory
    free(h_C);

    // Reset the device and exit
    cudaDeviceReset();

    bool success = true;
    return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
