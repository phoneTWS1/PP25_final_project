#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <chrono>

static int N;
static double *A, *B, *C_result, *C_true;

#define TILE 512  // 每個 tile 大小（必須為偶數）

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

void correctness_check(const double *C_true, const double *C_result, int N){
    int mismatch_count = 0;
    double tol = 5e-3 * (double)N * double(N);
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
}

// // CUDA Kernel: 矩陣加法/減法（支援 stride）
__global__ void add_matrix_stride_kernel(const double *a, int stride_a,
                                         const double *b, int stride_b,
                                         double *c, int stride_c,
                                         int n, double alpha, double beta){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < n && col < n){
        c[row * stride_c + col] = alpha * a[row * stride_a + col] + beta * b[row * stride_b + col];
    }
}

// CUDA Kernel: 標準矩陣乘法（支援 stride）
__global__ void matmul_stride_kernel(const double *a, int stride_a,
                                     const double *b, int stride_b,
                                     double *c, int stride_c,
                                     int n){
    __shared__ double As[32][32];
    __shared__ double Bs[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;

    double sum = 0.0;
    int numTiles = (n + 31) / 32;

    for(int t = 0; t < numTiles; t++){
        int aCol = t * 32 + tx;
        int bRow = t * 32 + ty;
        
        As[ty][tx] = a[row * stride_a + aCol];
        As[ty][tx] = 0.0;
        
        Bs[ty][tx] = b[bRow * stride_b + col];
        Bs[ty][tx] = 0.0;

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < 32; k++){
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if(row < n && col < n){
        c[row * stride_c + col] = sum;
    }
}

// 使用 Strassen 算法計算 tile 乘法：C_tile = A_tile * B_tile
// 直接使用 stride 存取 quadrants，不需要 cudaMemcpy2D
void strassen_tile_multiply(double *d_A_tile, double *d_B_tile, double *d_C_tile, 
                           int tile_size, 
                           double *d_work[], // 工作區陣列
                           cudaStream_t stream) {
    int n2 = tile_size / 2;  // quadrant 大小
    
    // 分配工作區索引
    double *d_M1 = d_work[0], *d_M2 = d_work[1], *d_M3 = d_work[2], *d_M4 = d_work[3];
    double *d_M5 = d_work[4], *d_M6 = d_work[5], *d_M7 = d_work[6];
    double *d_tmp[5] = {d_work[7], d_work[8], d_work[9], d_work[10], d_work[11]};
    
    // A 和 B 的四個 quadrants（使用指標偏移，stride = tile_size）
    double *A00 = d_A_tile;
    double *A01 = d_A_tile + n2;
    double *A10 = d_A_tile + n2 * tile_size;
    double *A11 = d_A_tile + n2 * tile_size + n2;
    
    double *B00 = d_B_tile;
    double *B01 = d_B_tile + n2;
    double *B10 = d_B_tile + n2 * tile_size;
    double *B11 = d_B_tile + n2 * tile_size + n2;
    
    // Kernel 參數
    dim3 block(16, 16);
    dim3 grid((n2 + 15) / 16, (n2 + 15) / 16);
    dim3 mul_block(32, 32);
    dim3 mul_grid((n2 + 31) / 32, (n2 + 31) / 32);
    
    // 計算 Strassen 的 7 個矩陣乘法
    
    // M1 = (A00 + A11) * (B00 + B11)
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(A00, tile_size, A11, tile_size, d_tmp[0], n2, n2, 1.0, 1.0);
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(B00, tile_size, B11, tile_size, d_tmp[1], n2, n2, 1.0, 1.0);
    matmul_stride_kernel<<<mul_grid, mul_block, 0, stream>>>(d_tmp[0], n2, d_tmp[1], n2, d_M1, n2, n2);
    
    // M2 = (A10 + A11) * B00
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(A10, tile_size, A11, tile_size, d_tmp[0], n2, n2, 1.0, 1.0);
    matmul_stride_kernel<<<mul_grid, mul_block, 0, stream>>>(d_tmp[0], n2, B00, tile_size, d_M2, n2, n2);
    
    // M3 = A00 * (B01 - B11)
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(B01, tile_size, B11, tile_size, d_tmp[0], n2, n2, 1.0, -1.0);
    matmul_stride_kernel<<<mul_grid, mul_block, 0, stream>>>(A00, tile_size, d_tmp[0], n2, d_M3, n2, n2);
    
    // M4 = A11 * (B10 - B00)
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(B10, tile_size, B00, tile_size, d_tmp[0], n2, n2, 1.0, -1.0);
    matmul_stride_kernel<<<mul_grid, mul_block, 0, stream>>>(A11, tile_size, d_tmp[0], n2, d_M4, n2, n2);
    
    // M5 = (A00 + A01) * B11
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(A00, tile_size, A01, tile_size, d_tmp[0], n2, n2, 1.0, 1.0);
    matmul_stride_kernel<<<mul_grid, mul_block, 0, stream>>>(d_tmp[0], n2, B11, tile_size, d_M5, n2, n2);
    
    // M6 = (A10 - A00) * (B00 + B01)
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(A10, tile_size, A00, tile_size, d_tmp[0], n2, n2, 1.0, -1.0);
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(B00, tile_size, B01, tile_size, d_tmp[1], n2, n2, 1.0, 1.0);
    matmul_stride_kernel<<<mul_grid, mul_block, 0, stream>>>(d_tmp[0], n2, d_tmp[1], n2, d_M6, n2, n2);
    
    // M7 = (A01 - A11) * (B10 + B11)
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(A01, tile_size, A11, tile_size, d_tmp[0], n2, n2, 1.0, -1.0);
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(B10, tile_size, B11, tile_size, d_tmp[1], n2, n2, 1.0, 1.0);
    matmul_stride_kernel<<<mul_grid, mul_block, 0, stream>>>(d_tmp[0], n2, d_tmp[1], n2, d_M7, n2, n2);
    
    // 組合結果計算 C 的四個 quadrants（直接寫入 d_C_tile）
    double *C00 = d_C_tile;
    double *C01 = d_C_tile + n2;
    double *C10 = d_C_tile + n2 * tile_size;
    double *C11 = d_C_tile + n2 * tile_size + n2;
    
    // C00 = M1 + M4 - M5 + M7
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(d_M1, n2, d_M4, n2, d_tmp[2], n2, n2, 1.0, 1.0);
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(d_tmp[2], n2, d_M5, n2, d_tmp[3], n2, n2, 1.0, -1.0);
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(d_tmp[3], n2, d_M7, n2, C00, tile_size, n2, 1.0, 1.0);
    
    // C01 = M3 + M5
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(d_M3, n2, d_M5, n2, C01, tile_size, n2, 1.0, 1.0);
    
    // C10 = M2 + M4
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(d_M2, n2, d_M4, n2, C10, tile_size, n2, 1.0, 1.0);
    
    // C11 = M1 - M2 + M3 + M6
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(d_M1, n2, d_M2, n2, d_tmp[2], n2, n2, 1.0, -1.0);
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(d_tmp[2], n2, d_M3, n2, d_tmp[3], n2, n2, 1.0, 1.0);
    add_matrix_stride_kernel<<<grid, block, 0, stream>>>(d_tmp[3], n2, d_M6, n2, C11, tile_size, n2, 1.0, 1.0);
}

// CUDA Kernel: 累加 tile 到全域矩陣
__global__ void accumulate_kernel(double *C_global, const double *C_tile,
                                  int C_row_offset, int C_col_offset, 
                                  int tile_size, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tile_size * tile_size;
    
    if(idx < total){
        int local_row = idx / tile_size;
        int local_col = idx % tile_size;
        int global_row = C_row_offset + local_row;
        int global_col = C_col_offset + local_col;
        
        if(global_row < N && global_col < N){
            C_global[global_row * N + global_col] += C_tile[idx];
        }
    }
}

int main(int argc, char **argv){
    if(argc != 5){
        fprintf(stderr, "usage: %s N A_filename B_filename C_true_filename\n", argv[0]);
        fprintf(stderr, "N must be divisible by TILE=%d\n", TILE);
        return 1;
    }

    N = atoi(argv[1]);
    const char *a_filename = argv[2];
    const char *b_filename = argv[3];
    const char *c_true_filename = argv[4];

    if(N <= 0 || (N % TILE) != 0 || (TILE % 2) != 0){
        fprintf(stderr, "Error: N=%d must be positive and divisible by TILE=%d. TILE must be even.\n", N, TILE);
        return 1;
    }

    printf("Loading data for N=%d with TILE=%d (Strassen inside tiles)...\n", N, TILE);
    load_matrix(&A, N, a_filename);
    load_matrix(&B, N, b_filename);
    load_matrix(&C_true, N, c_true_filename);

    C_result = (double*)calloc((size_t)N * N, sizeof(double));
    if (C_result == NULL) { 
        perror("Host malloc failed for C_result"); 
        cleanup_memory(); 
        return 1; 
    }

    int tiles = N / TILE;
    size_t tile_bytes = (size_t)TILE * TILE * sizeof(double);
    int n2 = TILE / 2;
    size_t quad_bytes = (size_t)n2 * n2 * sizeof(double);

    // 分配 device 記憶體
    double *d_A, *d_B, *d_C_global;
    double *d_A_tile, *d_B_tile, *d_C_tile;
    double *d_work[20]; // Strassen 工作區

    cudaMalloc(&d_A, (size_t)N * N * sizeof(double));
    cudaMalloc(&d_B, (size_t)N * N * sizeof(double));
    cudaMalloc(&d_C_global, (size_t)N * N * sizeof(double));
    cudaMalloc(&d_A_tile, tile_bytes);
    cudaMalloc(&d_B_tile, tile_bytes);
    cudaMalloc(&d_C_tile, tile_bytes);
    
    for(int i = 0; i < 20; i++){
        cudaMalloc(&d_work[i], quad_bytes);
    }

    // 拷貝輸入到 device
    cudaMemcpy(d_A, A, (size_t)N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (size_t)N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_C_global, 0, (size_t)N * N * sizeof(double));

    // 累加 kernel 參數
    int acc_threads = 1024;
    int acc_blocks = (TILE * TILE + acc_threads - 1) / acc_threads;

    // 創建 CUDA stream 用於重疊計算
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    printf("Computing tiled Strassen matrix multiplication...\n");

    // timing start (use std::chrono)
    auto chrono_start = std::chrono::steady_clock::now();
    
    // 主迴圈：分塊矩陣乘法
    // C[ti][tj] = Σ(k) A[ti][tk] * B[tk][tj]
    for(int ti = 0; ti < tiles; ++ti){
        for(int tj = 0; tj < tiles; ++tj){
            for(int tk = 0; tk < tiles; ++tk){
                // 從全域矩陣提取 tile
                int A_row_offset = ti * TILE;
                int A_col_offset = tk * TILE;
                int B_row_offset = tk * TILE;
                int B_col_offset = tj * TILE;
                
                cudaMemcpy2DAsync(d_A_tile, TILE * sizeof(double),
                           d_A + A_row_offset * N + A_col_offset, N * sizeof(double),
                           TILE * sizeof(double), TILE,
                           cudaMemcpyDeviceToDevice, stream);
                
                cudaMemcpy2DAsync(d_B_tile, TILE * sizeof(double),
                           d_B + B_row_offset * N + B_col_offset, N * sizeof(double),
                           TILE * sizeof(double), TILE,
                           cudaMemcpyDeviceToDevice, stream);
                
                // 使用 Strassen 計算 C_tile = A_tile * B_tile
                strassen_tile_multiply(d_A_tile, d_B_tile, d_C_tile, TILE, d_work, stream);
                
                // 累加到全域 C 矩陣
                int C_row_offset = ti * TILE;
                int C_col_offset = tj * TILE;
                accumulate_kernel<<<acc_blocks, acc_threads, 0, stream>>>(
                    d_C_global, d_C_tile, C_row_offset, C_col_offset, TILE, N);
            }
        }
    }

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }

    auto chrono_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = chrono_end - chrono_start;
    printf("Total computing time: %.6f s\n", elapsed.count());

    cudaMemcpy(C_result, d_C_global, (size_t)N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // timing end


    printf("Computation complete. Checking correctness...\n");
    correctness_check(C_true, C_result, N);

    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_global);
    cudaFree(d_A_tile);
    cudaFree(d_B_tile);
    cudaFree(d_C_tile);
    for(int i = 0; i < 20; i++){
        cudaFree(d_work[i]);
    }
    
    cleanup_memory();
    return 0;
}