#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<stdbool.h>

const int DIM = 3*32*32;
const int CLASSES = 10;
//const int N_TRAIN = 50000;
const int N_TRAIN = 500;
const int N_TEST = 10000;

__global__ void cuda_forward(float *x_ND, float *w_CD, float *o_NC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_TRAIN*CLASSES) {
        float sum = 0;
        int n = idx / CLASSES;
        int c = idx % CLASSES;
        for (int d = 0; d < DIM; d++) {
            sum += x_ND[n*DIM+d] * w_CD[c*DIM+d];
        }
        o_NC[idx] = sum;
    }
}

unsigned char* read_data(const char path[]) {

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    // Allocate a buffer to hold the file contents
    unsigned char *buffer = (unsigned char *)malloc(fileSize);
    if (buffer == NULL) {
        perror("Memory error");
        fclose(file);
        exit(1);
    }

    // Read the file into the buffer
    size_t bytesRead = fread(buffer, 1, fileSize, file);
    if (bytesRead != fileSize) {
        perror("Error reading file");
    }

    // Clean up
    fclose(file);

    return buffer;
}

float hash(float *x) {
    float sum = 0;
    for (int i = 0; i < 1000; i+=2) {
        sum += abs(x[i]);
    }
    return sum;
}

float *forward_linear(float *x_ND, float *w_CD, float *o_NC, int num) {
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < CLASSES; c++) {
            int idx = CLASSES * n + c;
            o_NC[idx] = 0;
            for (int d = 0; d < DIM; d++) {
                o_NC[idx] += x_ND[n*DIM+d] * w_CD[c*DIM+d];
            }
        }
    }
    return o_NC;
}

float *softmax(float *o_NC, int num) {
    float *z_NC = (float *)malloc(num*CLASSES*sizeof(float));
    for (int n = 0; n < num; n++) {
        float Z = 0;
        for (int c = 0; c < CLASSES; c++) {
            z_NC[n*CLASSES+c] = expf(o_NC[n*CLASSES+c]);
            Z += z_NC[n*CLASSES+c];
        }
        for (int c = 0; c < CLASSES; c++) {
            z_NC[n*CLASSES+c] /= Z;
        }
    }
    return z_NC;
}

float *one_hot(long *y, int num) {
    float *z = (float *)malloc(num*CLASSES*sizeof(float));
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < CLASSES; c++) {
            if (y[n] == c) {
                z[n*CLASSES+c] = 1;
            } else {
                z[n*CLASSES+c] = 0;
            }
        }
    }
    return z;
}

float *sub(float *x1, float *x2, int num) {
    for (int i = 0; i < num; i++) {
        x1[i] -= x2[i];
    }
    return x1;
}

float cross_entropy(float *p_NC, long *y_N, int num) {
    float loss = 0;
    for (int n = 0; n < num; n++) {
        if (!(y_N[n] >= 0 && y_N[n] < 10)) {
            printf("y[n] out of range: %ld\n", y_N[n]);
            exit(1);
        }
        loss += -logf(p_NC[n*CLASSES+y_N[n]]);
    }
    return loss;
}

const float ETA = 0.01/N_TRAIN;
float *fit_linear(float *x_ND, long *y_N) {

    cudaError_t err;
    size_t size;

    struct timespec start, end;
    double elapsed;

    float *xT_DN = (float *)malloc(N_TRAIN*DIM*sizeof(float));
    for (int n = 0; n < N_TRAIN; n++) {
        for (int d = 0; d < DIM; d++) {
            xT_DN[d*N_TRAIN+n] = x_ND[n*DIM+d];
        }
    }

    float *w_CD = (float *)malloc(CLASSES*DIM*sizeof(float));
    for (int c = 0; c < CLASSES; c++)
        for (int d = 0; d < DIM; d++)
            w_CD[c*DIM+d] = 0;

    float *xc_ND;
    size = N_TRAIN*DIM*sizeof(float);
    err = cudaMalloc((void**)&xc_ND, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(xc_ND, x_ND, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *wc_CD;
    size = CLASSES*DIM*sizeof(float);
    err = cudaMalloc((void**)&wc_CD, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(wc_CD, w_CD, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *oc_NC;
    size = N_TRAIN*CLASSES*sizeof(float);
    err = cudaMalloc((void**)&oc_NC, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *o_NC = (float *)malloc(size);

    int steps = 200;
    for (int step = 0; step < steps; step++) {

        clock_gettime(CLOCK_MONOTONIC, &start);

        int N = N_TRAIN*CLASSES;
        int threadsPerBlock = 1024;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        cuda_forward<<<blocksPerGrid, threadsPerBlock>>>(xc_ND, wc_CD, oc_NC);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        size = N_TRAIN*CLASSES*sizeof(float);
        err = cudaMemcpy(o_NC, oc_NC, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        float *p_NC = softmax(o_NC, N_TRAIN);
        float *delta_NC = sub(one_hot(y_N, N_TRAIN), p_NC, N_TRAIN*CLASSES);
        float loss = cross_entropy(p_NC, y_N, N_TRAIN);

        float *u_CD = (float *)malloc(CLASSES*DIM*sizeof(float));
        for (int c = 0; c < CLASSES; c++)
            for (int d = 0; d < DIM; d++)
                u_CD[c*DIM+d] = 0;

        for (int c = 0; c < CLASSES; c++) {
            for (int d = 0; d < DIM; d++) {
                for (int n = 0; n < N_TRAIN; n++) {
                    u_CD[c*DIM+d] += delta_NC[n*CLASSES+c] * xT_DN[d*N_TRAIN+n];
                    //u_CD[c*DIM+d] += delta_NC[n*CLASSES+c] * x_ND[n*DIM+d]; // 6x slower bc of bad memory access pattern
                }
            }
        }
        for (int c = 0; c < CLASSES; c++)
            for (int d = 0; d < DIM; d++)
                w_CD[c*DIM+d] += ETA * u_CD[c*DIM+d];

        size = CLASSES*DIM*sizeof(float);
        err = cudaMemcpy(wc_CD, w_CD, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy vector C from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        free(u_CD);

        free(p_NC);
        free(delta_NC);

        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - start.tv_sec);
        elapsed += (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Time elapsed: %.6f seconds\n", elapsed);

        printf("Step: %d, Loss: %f\n", step, loss/N_TRAIN);
    }


    free(xT_DN);
    free(o_NC);
    cudaFree(xc_ND);
    cudaFree(wc_CD);
    cudaFree(oc_NC);

    return w_CD;
}

int eval_linear(float *w_CD, float *x_MD, long *y_M) {
    float *o_MC = (float *)malloc(N_TEST*CLASSES*sizeof(float));
    forward_linear(x_MD, w_CD, o_MC, N_TEST);
    int correct = 0;
    for (int m = 0; m < N_TEST; m++) {
        int max_i = 0;
        int max_v = -100000;
        for (int c = 0; c < CLASSES; c++) {
            if (o_MC[m*CLASSES+c] > max_v) {
                max_i = c;
                max_v = o_MC[m*CLASSES+c];
            }
        }
        if (max_i == y_M[m]) {
            correct += 1;
        }
    }
    free(o_MC);
    return correct;
}

int main() {
    float *train_x_ND = (float *)read_data("/home/ubuntu/notebooks/train_x.bin");
    long *train_y_N = (long *)read_data("/home/ubuntu/notebooks/train_y.bin");
    float *weight_CD = fit_linear(train_x_ND, train_y_N);
    free(train_x_ND);
    free(train_y_N);

    float *test_x_MD = (float *)read_data("/home/ubuntu/notebooks/test_x.bin");
    long *test_y_M = (long *)read_data("/home/ubuntu/notebooks/test_y.bin");

    int correct = eval_linear(weight_CD, test_x_MD, test_y_M);
    printf("Correct: %d\n", correct);

    free(test_x_MD);
    free(test_y_M);
    free(weight_CD);

    cudaDeviceReset();

    return 0;
}
