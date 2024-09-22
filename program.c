#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<stdbool.h>

unsigned char* read_data(char *path) {

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

const int DIM = 3*32*32;
const int CLASSES = 10;
//const int N_TRAIN = 50000;
const int N_TRAIN = 1000;
const int N_TEST = 10000;

float *forward_linear(float *x_ND, float *w_CD, int num) {
    float *o_NC = (float *)malloc(num*CLASSES*sizeof(float));
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

float *fit_linear(float *x_ND, long *y_N) {

    float eta = 0.03/N_TRAIN;

    // Allocate weight: 10 x (3x32x32)
    float *w_CD = (float *)malloc(CLASSES*DIM*sizeof(float));
    for (int c = 0; c < CLASSES; c++)
        for (int d = 0; d < DIM; d++)
            w_CD[c*DIM+d] = 0;

    int steps = 200;
    for (int step = 0; step < steps; step++) {

        struct timespec start, end;
        double elapsed;
        clock_gettime(CLOCK_MONOTONIC, &start);

        float *o_NC = forward_linear(x_ND, w_CD, N_TRAIN);
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
                    u_CD[c*DIM+d] += delta_NC[n*CLASSES+c] * x_ND[n*DIM+d];
                }
            }
        }
        for (int c = 0; c < CLASSES; c++)
            for (int d = 0; d < DIM; d++)
                w_CD[c*DIM+d] += eta * u_CD[c*DIM+d];

        free(o_NC);
        free(p_NC);
        free(delta_NC);
        free(u_CD);

        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - start.tv_sec);
        elapsed += (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Step: %d, Loss: %f, ", step, loss/N_TRAIN);
        printf("Time elapsed: %.6f seconds\n", elapsed);
    }

    return w_CD;
}

int eval_linear(float *w_CD, float *x_MD, long *y_M) {
    float *o_MC = forward_linear(x_MD, w_CD, N_TEST);
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
    return correct;
}

int main() {
    float *train_x_ND = (float *) read_data("/home/ubuntu/notebooks/train_x.bin");
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
    return 0;
}
