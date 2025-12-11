#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int N;
static double *A, *B, *C_result, *C_true;
static double *A00, *A01, *A10, *A11;
static double *B00, *B01, *B10, *B11;

void load_matrix(double **mat_ptr, int N_dim, const char *filename) {
    long long size = (long long)N_dim * N_dim;
    size_t bytes_f = size * sizeof(float);
    size_t bytes_d = size * sizeof(double);

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // read as float then convert to double (dataset produced float32)
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

void divide(int n2){

    A00 = (double*)calloc(n2 * n2, sizeof(double));
    A01 = (double*)calloc(n2 * n2, sizeof(double));
    A10 = (double*)calloc(n2 * n2, sizeof(double));
    A11 = (double*)calloc(n2 * n2, sizeof(double));
    B00 = (double*)calloc(n2 * n2, sizeof(double));
    B01 = (double*)calloc(n2 * n2, sizeof(double));
    B10 = (double*)calloc(n2 * n2, sizeof(double));
    B11 = (double*)calloc(n2 * n2, sizeof(double));

    size_t n2row = n2 * sizeof(double);
    for(int i=0; i<n2; i++){
        memcpy(A00 + i*n2, A + i*N, n2row);
        memcpy(A01 + i*n2, A + i*N + n2, n2row);
        memcpy(A10 + i*n2, A + (n2 + i)*N, n2row);
        memcpy(A11 + i*n2, A + (n2 + i)*N + n2, n2row);

        memcpy(B00 + i*n2, B + i*N, n2row);
        memcpy(B01 + i*n2, B + i*N + n2, n2row);
        memcpy(B10 + i*n2, B + (n2 + i)*N, n2row);
        memcpy(B11 + i*n2, B + (n2 + i)*N + n2, n2row);
    }
}

void merge(int n2){
    size_t n2row = n2 * sizeof(double);
    for(int i=0; i<n2; i++){
        memcpy(C_result + i*N, A00 + i*n2, n2row);
        memcpy(C_result + i*N + n2, A01 + i*n2, n2row);
        memcpy(C_result + (n2 + i)*N, A10 + i*n2, n2row);
        memcpy(C_result + (n2 + i)*N + n2, A11 + i*n2, n2row);
    }
}

// CUDA Kernel: 矩陣加法/減法 (double)
__global__ void add_matrix_kernel(const double *a, const double *b, double *c, int n, double alpha, double beta){
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n * n;

    if(idx < total){
        c[idx] = alpha * a[idx] + beta * b[idx];
    }
}

// CUDA Kernel: Tiled 矩陣乘法 (double)
__global__ void matmul_kernel(const double *a, const double *b, double *c, int n){
    __shared__ double As[32][32];
    __shared__ double Bs[32][32];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * 32 + ty;
    int col = bx * 32 + tx;

    double sum = 0.0;

    int numTiles = (n + 31) / 32;

    for(int t = 0; t < numTiles; t++){
        // 載入 A 的 tile
        int aRow = row;
        int aCol = t * 32 + tx;
        if(aRow < n && aCol < n){
            As[ty][tx] = a[aRow * n + aCol];
        } else {
            As[ty][tx] = 0.0;
        }

        // 載入 B 的 tile
        int bRow = t * 32 + ty;
        int bCol = col;
        if(bRow < n && bCol < n){
            Bs[ty][tx] = b[bRow * n + bCol];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        // 計算
        #pragma unroll
        for(int k = 0; k < 32; k++){
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 寫回結果
    if(row < n && col < n){
        c[row * n + col] = sum;
    }
}

void correctness_check(const double *C_true, const double *C_result, int N){
    int mismatch_count = 0;
    double tol = 5e-4 * (double)N;
    long long sz = (long long)N * N;
    double max_error = 0.0;

    for(long long i=0; i < sz; i++){
        double error = fabs(C_result[i] - C_true[i]);
        if (error > max_error) max_error = error;

        if (error > tol){
             mismatch_count++;
             if (mismatch_count <= 10) {
                 fprintf(stderr, "Mismatch at index %lld: Result=%.6f, True=%.6f, Error=%.6f\n",
                         i, C_result[i], C_true[i], error);
             }
        }
    }

    printf("Maximum error: %.6f\n", max_error);

    if(mismatch_count > 0){
        printf("FAILED: Result mismatch with loaded answer C. Total errors: %d\n", mismatch_count);
    } else {
        printf("SUCCESS: Result matches loaded answer C.\n");
    }
}

void cleanup_memory() {
    if (A) free(A);
    if (B) free(B);
    if (C_true) free(C_true);
    if (C_result) free(C_result);

    if (A00) free(A00); if (A01) free(A01); if (A10) free(A10); if (A11) free(A11);
    if (B00) free(B00); if (B01) free(B01); if (B10) free(B10); if (B11) free(B11);
}

int main(int argc, char **argv){
    if(argc != 5){
        fprintf(stderr, "usage: %s N A_filename B_filename C_true_filename\n", argv[0]);
        fprintf(stderr, "N must be even for 1-level Strassen\n");
        return 1;
    }

    N = atoi(argv[1]);
    const char *a_filename = argv[2];
    const char *b_filename = argv[3];
    const char *c_true_filename = argv[4];

    if(N <= 0 || (N % 2) != 0){
        fprintf(stderr, "Error: N=%d must be positive and even.\n", N);
        return 1;
    }

    printf("Loading data for N=%d...\n", N);
    load_matrix(&A, N, a_filename);
    load_matrix(&B, N, b_filename);
    load_matrix(&C_true, N, c_true_filename);

    C_result = (double*)calloc((size_t)N * N, sizeof(double));
    if (C_result == NULL) {
        perror("Host malloc failed for C_result");
        cleanup_memory();
        return 1;
    }

    int n2 = N/2;
    divide(n2);

    // 分配 GPU 記憶體 (double)
    double *d_A00, *d_A01, *d_A10, *d_A11;
    double *d_B00, *d_B01, *d_B10, *d_B11;
    double *d_m1, *d_m2, *d_m3, *d_m4, *d_m5, *d_m6, *d_m7;
    double *d_C00, *d_C01, *d_C10, *d_C11;
    double *d_tmp[10];

    size_t bytes = (size_t)n2 * n2 * sizeof(double);

    cudaMalloc(&d_A00, bytes); cudaMalloc(&d_A01, bytes);
    cudaMalloc(&d_A10, bytes); cudaMalloc(&d_A11, bytes);
    cudaMalloc(&d_B00, bytes); cudaMalloc(&d_B01, bytes);
    cudaMalloc(&d_B10, bytes); cudaMalloc(&d_B11, bytes);
    cudaMalloc(&d_m1, bytes); cudaMalloc(&d_m2, bytes);
    cudaMalloc(&d_m3, bytes); cudaMalloc(&d_m4, bytes);
    cudaMalloc(&d_m5, bytes); cudaMalloc(&d_m6, bytes);
    cudaMalloc(&d_m7, bytes);
    cudaMalloc(&d_C00, bytes); cudaMalloc(&d_C01, bytes);
    cudaMalloc(&d_C10, bytes); cudaMalloc(&d_C11, bytes);

    for(int i = 0; i < 10; i++){
        cudaMalloc(&d_tmp[i], bytes);
    }

    // 複製資料到 GPU (注意 host sub-blocks are double)
    cudaMemcpy(d_A00, A00, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A01, A01, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A10, A10, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A11, A11, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B00, B00, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B01, B01, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B10, B10, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B11, B11, bytes, cudaMemcpyHostToDevice);

    // 設定 kernel 參數
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n2 * n2 + threadsPerBlock - 1) / threadsPerBlock;

    dim3 matmul_block(32, 32);
    dim3 matmul_grid((n2 + 31) / 32, (n2 + 31) / 32);

    // 計時開始
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Strassen 的 10 個加減法 (double constants)
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A00, d_A11, d_tmp[0], n2, 1.0, 1.0);  // S1
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B00, d_B11, d_tmp[1], n2, 1.0, 1.0);  // S2
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A10, d_A11, d_tmp[2], n2, 1.0, 1.0);  // S3
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B01, d_B11, d_tmp[3], n2, 1.0, -1.0); // S4
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B10, d_B00, d_tmp[4], n2, 1.0, -1.0); // S5
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A00, d_A01, d_tmp[5], n2, 1.0, 1.0);  // S6
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A10, d_A00, d_tmp[6], n2, 1.0, -1.0); // S7
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B00, d_B01, d_tmp[7], n2, 1.0, 1.0);  // S8
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A01, d_A11, d_tmp[8], n2, 1.0, -1.0); // S9
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B10, d_B11, d_tmp[9], n2, 1.0, 1.0);  // S10

    // 7 次矩陣乘法 (double)
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_tmp[0], d_tmp[1], d_m1, n2); // M1
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_tmp[2], d_B00, d_m2, n2);    // M2
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_A00, d_tmp[3], d_m3, n2);    // M3
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_A11, d_tmp[4], d_m4, n2);    // M4
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_tmp[5], d_B11, d_m5, n2);    // M5
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_tmp[6], d_tmp[7], d_m6, n2); // M6
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_tmp[8], d_tmp[9], d_m7, n2); // M7

    // 組合結果 (double)
    // C00 = M1 + M4 - M5 + M7  -> implemented via add kernels
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m1, d_m7, d_tmp[1], n2, 1.0, 1.0);
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m4, d_m5, d_tmp[2], n2, 1.0, -1.0);
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tmp[1], d_tmp[2], d_C00, n2, 1.0, 1.0);

    // C01 = M3 + M5
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m3, d_m5, d_C01, n2, 1.0, 1.0);

    // C10 = M2 + M4
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m2, d_m4, d_C10, n2, 1.0, 1.0);

    // C11 = M1 - M2 + M3 + M6
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m1, d_m2, d_tmp[4], n2, 1.0, -1.0);
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m3, d_m6, d_tmp[5], n2, 1.0, 1.0);
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tmp[4], d_tmp[5], d_C11, n2, 1.0, 1.0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Compute time: %.6f s\n", milliseconds / 1000.0f);

    // 檢查 CUDA 錯誤
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 複製結果回 Host (double)
    cudaMemcpy(A00, d_C00, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(A01, d_C01, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(A10, d_C10, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(A11, d_C11, bytes, cudaMemcpyDeviceToHost);

    merge(n2);
    correctness_check(C_true, C_result, N);

    // 清理 GPU 記憶體
    cudaFree(d_A00); cudaFree(d_A01); cudaFree(d_A10); cudaFree(d_A11);
    cudaFree(d_B00); cudaFree(d_B01); cudaFree(d_B10); cudaFree(d_B11);
    cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_m3); cudaFree(d_m4);
    cudaFree(d_m5); cudaFree(d_m6); cudaFree(d_m7);
    cudaFree(d_C00); cudaFree(d_C01); cudaFree(d_C10); cudaFree(d_C11);
    for(int i = 0; i < 10; i++) cudaFree(d_tmp[i]);

    cleanup_memory();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}