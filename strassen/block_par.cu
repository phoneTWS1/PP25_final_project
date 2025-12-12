#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

static int N;
static double *A = nullptr, *B = nullptr, *C_result = nullptr, *C_true = nullptr;

#define BS 32  // thread block dimensions fixed to 32x32
int TILE = 512;

static void checkCuda(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

void load_matrix(double **mat_ptr, int N_dim, const char *filename) {
    long long size = (long long)N_dim * N_dim;
    size_t bytes_f = size * sizeof(float);
    size_t bytes_d = size * sizeof(double);

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    float *tmp = (float *)malloc(bytes_f);
    if (tmp == NULL) {
        perror("malloc failed for tmp float buffer");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    size_t read_count = fread(tmp, sizeof(float), size, file);
    if (read_count != size) {
        fprintf(stderr, "Error: Read incomplete from %s. Expected %lld elements, read %zu.\n",
                filename, size, read_count);
        free(tmp);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);

    *mat_ptr = (double *)malloc(bytes_d);
    if (*mat_ptr == NULL) {
        perror("Host malloc failed in load_matrix (double)");
        free(tmp);
        exit(EXIT_FAILURE);
    }

    for (long long i = 0; i < size; ++i) {
        (*mat_ptr)[i] = (double)tmp[i];
    }
    free(tmp);
}

void correctness_check(const double *C_true_loc, const double *C_result_loc, int N_loc){
    long long sz = (long long)N_loc * N_loc;
    double max_error = 0.0;
    long long mismatches = 0;
    double tol = 5e-3 * double(N) * double(N); // 可根據需求調整
    for (long long i = 0; i < sz; ++i) {
        double err = fabs(C_result_loc[i] - C_true_loc[i]);
        if (err > max_error) max_error = err;
        if (err > tol) {
            mismatches++;
            if (mismatches <= 10) {
                fprintf(stderr, "Mismatch idx %lld: res=%.8f true=%.8f err=%.8f\n", i, C_result_loc[i], C_true_loc[i], err);
            }
        }
    }
    printf("max error = %.8e, mismatches = %lld\n", max_error, mismatches);
    if (mismatches == 0) printf("SUCCESS\n");
    else printf("FAILED\n");
}

void cleanup_memory() {
    if (A) free(A);
    if (B) free(B);
    if (C_true) free(C_true);
    if (C_result) free(C_result);
}

__global__ void matmul_block_kernel(const double *A, const double *B, double *C, int N, int TILE) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // [0,BS)
    int ty = threadIdx.y; // [0,BS)

    // repetition factor R = TILE / BS (assume TILE % BS == 0)
    int R = TILE / BS;

    // base coordinates for this block tile
    int row_base = by * TILE;
    int col_base = bx * TILE;

    // for each sub-output (ii, jj) inside the TILE that this thread is responsible for:
    for (int ii = 0; ii < R; ++ii) {
        int row = row_base + ii * BS + ty; // concrete global row
        if (row >= N) continue;
        for (int jj = 0; jj < R; ++jj) {
            int col = col_base + jj * BS + tx; // concrete global col
            if (col >= N) continue;

            double sum = 0.0;
            // standard dot-product for C[row, col]
            for (int k = 0; k < N; ++k) {
                double a = A[(long long)row * N + k];
                double b = B[(long long)k * N + col];
                sum += a * b;
            }
            C[(long long)row * N + col] = sum;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 6) {
        fprintf(stderr, "usage: %s N A_filename B_filename C_true_filename [TILE]\n", argv[0]);
        fprintf(stderr, "  TILE (optional) = output tile size, must be multiple of %d. default 64\n", BS);
        return 1;
    }

    N = atoi(argv[1]);
    const char *a_filename = argv[2];
    const char *b_filename = argv[3];
    const char *c_true_filename = argv[4];
    if (argc == 6) TILE = atoi(argv[5]);

    if (N <= 0) {
        fprintf(stderr, "N must be positive\n");
        return 1;
    }
    if (TILE < BS) TILE = BS;
    if ((TILE % BS) != 0) {
        fprintf(stderr, "TILE (%d) must be multiple of BS (%d)\n", TILE, BS);
        return 1;
    }

    printf("Loading data N=%d TILE=%d ...\n", N, TILE);
    load_matrix(&A, N, a_filename);
    load_matrix(&B, N, b_filename);
    load_matrix(&C_true, N, c_true_filename);

    C_result = (double*)calloc((size_t)N * N, sizeof(double));
    if (!C_result) { perror("calloc C_result"); cleanup_memory(); return 1; }

    // device buffers
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t bytes = (size_t)N * N * sizeof(double);
    checkCuda(cudaMalloc((void**)&d_A, bytes), "malloc d_A");
    checkCuda(cudaMalloc((void**)&d_B, bytes), "malloc d_B");
    checkCuda(cudaMalloc((void**)&d_C, bytes), "malloc d_C");

    checkCuda(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice), "memcpy H2D A");
    checkCuda(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice), "memcpy H2D B");
    checkCuda(cudaMemset(d_C, 0, bytes), "memset d_C");

    dim3 block(BS, BS);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // timing with chrono (includes kernel execution and sync)
    auto t0 = std::chrono::steady_clock::now();

    matmul_block_kernel<<<grid, block>>>(d_A, d_B, d_C, N, TILE);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "device synchronize");

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    printf("Compute time: %.6f s\n", elapsed.count());

    checkCuda(cudaMemcpy(C_result, d_C, bytes, cudaMemcpyDeviceToHost), "memcpy D2H C");

    printf("Checking correctness...\n");
    correctness_check(C_true, C_result, N);

    // cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cleanup_memory();
    return 0;
}