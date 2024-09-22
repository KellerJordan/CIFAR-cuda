#include<stdio.h>
#include<stdlib.h>
#include<time.h>

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
    // 50000x3x32x32, 3x32x32x10
    float *o = (float *)malloc(num*CLASSES);
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

float *fit_linear(float *x, long *y) {
    // 50000x3x32x32, 50000
    float *w = (float *)malloc(CLASSES*DIM*4);

    int steps = 100;
    for (int s = 0; s < steps; s++) {

        struct timespec start, end;
        double elapsed;
        clock_gettime(CLOCK_MONOTONIC, &start);

        float *o = forward_linear(x, w, N_TRAIN);
        printf("done\n");

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
    float *test_x = (float *)read_data("/home/ubuntu/notebooks/test_x.bin");
    long *test_y = (long *)read_data("/home/ubuntu/notebooks/test_y.bin");
    // 50000x3x32x32, 10000x3x32x32

    float *weight = fit_linear(train_x, train_y);

    /*
    printf("%f\n", img.data[0]);
    printf("%f\n", img.data[1]);
    printf("size: %ld\n", img.size);
    */
    printf("%f\n", test_x[0]);
    printf("%f\n", weight[0]);

    free(train_x);
    free(train_y);
    free(test_x);
    free(test_y);
    free(weight);
    return 0;
}
