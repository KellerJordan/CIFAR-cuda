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
const int N_TRAIN = 50000;
const int N_TEST = 10000;

float *forward_linear(float *x, float *w, int num) {
    float *o = (float *)malloc(num*CLASSES*sizeof(float));
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < CLASSES; c++) {
            int idx = CLASSES * n + c;
            o[idx] = 0;
            for (int i = 0; i < DIM; i++) {
                o[idx] += x[i] * w[DIM*c+i];
            }
        }
    }
    return o;
}

float *softmax(float *o, int num) {
    float *z = (float *)malloc(num*CLASSES*sizeof(float));
    for (int n = 0; n < num; n++) {
        float Z = 0;
        for (int c = 0; c < CLASSES; c++) {
            z[n*CLASSES+c] = expf(o[n*CLASSES+c]);
            Z += z[n*CLASSES+c];
        }
        for (int c = 0; c < CLASSES; c++) {
            z[n*CLASSES+c] /= Z;
        }
    }
    return z;
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

float cross_entropy(float *p, long *y, int num) {
    float loss = 0;
    for (int n = 0; n < num; n++) {
        if (!(y[n] >= 0 && y[n] < 10)) {
            printf("y[n] out of range: %ld\n", y[n]);
            exit(1);
        }
        loss += -logf(p[n*CLASSES+y[n]]);
    }
    return loss;
}

float *fit_linear(float *x, long *y) {

    float eta = 0.05/50000.;

    // Allocate weight: 10 x (3x32x32)
    float *w = (float *)malloc(CLASSES*DIM*sizeof(float));
    for (int c = 0; c < CLASSES; c++)
        for (int d = 0; d < DIM; d++)
            w[c*DIM+d] = 0;

    int steps = 100;
    for (int step = 0; step < steps; step++) {

        struct timespec start, end;
        double elapsed;
        clock_gettime(CLOCK_MONOTONIC, &start);

        float *o = forward_linear(x, w, N_TRAIN);
        float *p = softmax(o, N_TRAIN);
        float *delta = sub(one_hot(y, N_TRAIN), p, N_TRAIN*CLASSES);
        float loss = cross_entropy(p, y, N_TRAIN);
        printf("Step: %d, Loss: %f\n", step, loss/N_TRAIN);

        printf("weight: %f\n", w[0]);
        /*
        for (int c = 0; c < 10; c++) {
            printf("%f  ", delta[c]);
        }
        printf("\n");
        */

        for (int c = 0; c < CLASSES; c++) {
            for (int d = 0; d < DIM; d++) {
                for (int n = 0; n < N_TRAIN; n++) {
                    w[c*DIM+d] += eta * delta[n*CLASSES+c] * x[n*DIM+d];
                }
            }
        }

        free(o);
        free(p);
        free(delta);

        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - start.tv_sec);
        elapsed += (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Time elapsed: %.9f seconds\n", elapsed);
    }

    return w;
}

int main() {
    float *train_x = (float *) read_data("/home/ubuntu/notebooks/train_x.bin");
    long *train_y = (long *)read_data("/home/ubuntu/notebooks/train_y.bin");
    float *weight = fit_linear(train_x, train_y);
    free(train_x);
    free(train_y);

    float *test_x = (float *)read_data("/home/ubuntu/notebooks/test_x.bin");
    long *test_y = (long *)read_data("/home/ubuntu/notebooks/test_y.bin");

    free(test_x);
    free(test_y);
    free(weight);
    return 0;
}
