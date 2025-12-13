#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <chrono>

static int N;
static double *A, *B, *C_result, *C_true;

#define TILE 512
#define NUM_STREAMS 20

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

// ============================================================================
// 優化的矩陣加減法 kernel - 直接從源矩陣讀取指定區域
// ============================================================================
__global__ void matrix_add_sub_kernel(
    const double *src1, int src1_row_offset, int src1_col_offset, int src1_stride,
    const double *src2, int src2_row_offset, int src2_col_offset, int src2_stride,
    double *dst, int n, int op)  // op: 0=add, 1=sub
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    if(idx < total) {
        int row = idx / n;
        int col = idx % n;
        
        double val1 = src1[(src1_row_offset + row) * src1_stride + (src1_col_offset + col)];
        double val2 = src2[(src2_row_offset + row) * src2_stride + (src2_col_offset + col)];
        
        dst[idx] = (op == 0) ? (val1 + val2) : (val1 - val2);
    }
}

// ============================================================================
// 優化的矩陣乘法 kernel - 直接從源矩陣讀取指定區域
// 支援從大矩陣中讀取任意位置的 tile
// ============================================================================
__global__ void matmul_kernel_offset(
    const double *a, int a_row_offset, int a_col_offset, int a_stride,
    const double *b, int b_row_offset, int b_col_offset, int b_stride,
    double *c, int n)
{
    __shared__ double As[32][32];
    __shared__ double Bs[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;

    double sum = 0.0;
    int numTiles = (n + 31) / 32;

    for(int t = 0; t < numTiles; t++){
        // Load A tile with offset
        int aRow = a_row_offset + row;
        int aCol = a_col_offset + t * 32 + tx;
        if(row < n && t * 32 + tx < n){
            As[ty][tx] = a[aRow * a_stride + aCol];
        } else {
            As[ty][tx] = 0.0;
        }

        // Load B tile with offset
        int bRow = b_row_offset + t * 32 + ty;
        int bCol = b_col_offset + col;
        if(t * 32 + ty < n && col < n){
            Bs[ty][tx] = b[bRow * b_stride + bCol];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < 32; k++){
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if(row < n && col < n){
        c[row * n + col] = sum;
    }
}

// ============================================================================
// Kernel Fusion: 合併多個矩陣加減法操作
// ============================================================================
__global__ void fused_strassen_prep_kernel(
    const double *A, int A_stride, int a00_r, int a00_c, int a01_r, int a01_c,
    int a10_r, int a10_c, int a11_r, int a11_c,
    const double *B, int B_stride, int b00_r, int b00_c, int b01_r, int b01_c,
    int b10_r, int b10_c, int b11_r, int b11_c,
    double *S1, double *S2, double *S3, double *S4,
    double *T1, double *T2, double *T3, double *T4,
    int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    if(idx < total) {
        int row = idx / n;
        int col = idx % n;
        
        // 直接從 A 讀取四個 quadrants
        double a00 = A[(a00_r + row) * A_stride + (a00_c + col)];
        double a01 = A[(a01_r + row) * A_stride + (a01_c + col)];
        double a10 = A[(a10_r + row) * A_stride + (a10_c + col)];
        double a11 = A[(a11_r + row) * A_stride + (a11_c + col)];
        
        // 直接從 B 讀取四個 quadrants
        double b00 = B[(b00_r + row) * B_stride + (b00_c + col)];
        double b01 = B[(b01_r + row) * B_stride + (b01_c + col)];
        double b10 = B[(b10_r + row) * B_stride + (b10_c + col)];
        double b11 = B[(b11_r + row) * B_stride + (b11_c + col)];
        
        // 計算中間結果
        S1[idx] = a00 + a11;
        S2[idx] = a10 + a11;
        S3[idx] = a00 + a01;
        S4[idx] = a10 - a00;
        
        T1[idx] = b00 + b11;
        T2[idx] = b01 - b11;
        T3[idx] = b10 - b00;
        T4[idx] = b00 + b01;
    }
}

// ============================================================================
// Kernel Fusion: M7 準備
// ============================================================================
__global__ void fused_m7_prep_kernel(
    const double *A, int A_stride, int a01_r, int a01_c, int a11_r, int a11_c,
    const double *B, int B_stride, int b10_r, int b10_c, int b11_r, int b11_c,
    double *S5, double *T5, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    if(idx < total) {
        int row = idx / n;
        int col = idx % n;
        
        double a01 = A[(a01_r + row) * A_stride + (a01_c + col)];
        double a11 = A[(a11_r + row) * A_stride + (a11_c + col)];
        double b10 = B[(b10_r + row) * B_stride + (b10_c + col)];
        double b11 = B[(b11_r + row) * B_stride + (b11_c + col)];
        
        S5[idx] = a01 - a11;
        T5[idx] = b10 + b11;
    }
}

// ============================================================================
// Kernel Fusion: 合併結果
// ============================================================================
__global__ void fused_strassen_combine_kernel(
    const double *M1, const double *M2, const double *M3, 
    const double *M4, const double *M5, const double *M6, const double *M7,
    double *C, int C_stride, int c00_r, int c00_c, int c01_r, int c01_c,
    int c10_r, int c10_c, int c11_r, int c11_c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    if(idx < total) {
        int row = idx / n;
        int col = idx % n;
        
        // C00 = M1 + M4 - M5 + M7
        C[(c00_r + row) * C_stride + (c00_c + col)] = 
            M1[idx] + M4[idx] - M5[idx] + M7[idx];
        
        // C01 = M3 + M5
        C[(c01_r + row) * C_stride + (c01_c + col)] = 
            M3[idx] + M5[idx];
        
        // C10 = M2 + M4
        C[(c10_r + row) * C_stride + (c10_c + col)] = 
            M2[idx] + M4[idx];
        
        // C11 = M1 - M2 + M3 + M6
        C[(c11_r + row) * C_stride + (c11_c + col)] = 
            M1[idx] - M2[idx] + M3[idx] + M6[idx];
    }
}

// ============================================================================
// 累加 kernel
// ============================================================================
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

// ============================================================================
// Strassen tile multiply - 直接從大矩陣讀取，無需 cudaMemcpy2D
// ============================================================================
void strassen_tile_multiply_direct(
    double *d_A, int A_row_offset, int A_col_offset, int A_stride,
    double *d_B, int B_row_offset, int B_col_offset, int B_stride,
    double *d_C, int C_row_offset, int C_col_offset, int C_stride,
    int tile_size, 
    double *d_work[],
    cudaStream_t stream) 
{
    int n2 = tile_size / 2;
    
    // 工作區
    double *d_M1 = d_work[0], *d_M2 = d_work[1], *d_M3 = d_work[2], *d_M4 = d_work[3];
    double *d_M5 = d_work[4], *d_M6 = d_work[5], *d_M7 = d_work[6];
    double *d_S1 = d_work[7], *d_S2 = d_work[8], *d_S3 = d_work[9];
    double *d_S4 = d_work[10], *d_S5 = d_work[11];
    double *d_T1 = d_work[12], *d_T2 = d_work[13], *d_T3 = d_work[14];
    double *d_T4 = d_work[15], *d_T5 = d_work[16];
    
    int threads = 256;
    int blocks = (n2 * n2 + threads - 1) / threads;
    dim3 mul_block(32, 32);
    dim3 mul_grid((n2 + 31) / 32, (n2 + 31) / 32);
    
    // 計算四個 quadrants 的偏移量
    int a00_r = A_row_offset, a00_c = A_col_offset;
    int a01_r = A_row_offset, a01_c = A_col_offset + n2;
    int a10_r = A_row_offset + n2, a10_c = A_col_offset;
    int a11_r = A_row_offset + n2, a11_c = A_col_offset + n2;
    
    int b00_r = B_row_offset, b00_c = B_col_offset;
    int b01_r = B_row_offset, b01_c = B_col_offset + n2;
    int b10_r = B_row_offset + n2, b10_c = B_col_offset;
    int b11_r = B_row_offset + n2, b11_c = B_col_offset + n2;
    
    // 準備 M1-M6 的輸入
    fused_strassen_prep_kernel<<<blocks, threads, 0, stream>>>(
        d_A, A_stride, a00_r, a00_c, a01_r, a01_c, a10_r, a10_c, a11_r, a11_c,
        d_B, B_stride, b00_r, b00_c, b01_r, b01_c, b10_r, b10_c, b11_r, b11_c,
        d_S1, d_S2, d_S3, d_S4, d_T1, d_T2, d_T3, d_T4, n2
    );
    
    // 計算 M1-M6
    matmul_kernel_offset<<<mul_grid, mul_block, 0, stream>>>(
        d_S1, 0, 0, n2, d_T1, 0, 0, n2, d_M1, n2);
    matmul_kernel_offset<<<mul_grid, mul_block, 0, stream>>>(
        d_S2, 0, 0, n2, d_B, b00_r, b00_c, B_stride, d_M2, n2);
    matmul_kernel_offset<<<mul_grid, mul_block, 0, stream>>>(
        d_A, a00_r, a00_c, A_stride, d_T2, 0, 0, n2, d_M3, n2);
    matmul_kernel_offset<<<mul_grid, mul_block, 0, stream>>>(
        d_A, a11_r, a11_c, A_stride, d_T3, 0, 0, n2, d_M4, n2);
    matmul_kernel_offset<<<mul_grid, mul_block, 0, stream>>>(
        d_S3, 0, 0, n2, d_B, b11_r, b11_c, B_stride, d_M5, n2);
    matmul_kernel_offset<<<mul_grid, mul_block, 0, stream>>>(
        d_S4, 0, 0, n2, d_T4, 0, 0, n2, d_M6, n2);
    
    // 準備 M7
    fused_m7_prep_kernel<<<blocks, threads, 0, stream>>>(
        d_A, A_stride, a01_r, a01_c, a11_r, a11_c,
        d_B, B_stride, b10_r, b10_c, b11_r, b11_c,
        d_S5, d_T5, n2
    );
    
    matmul_kernel_offset<<<mul_grid, mul_block, 0, stream>>>(
        d_S5, 0, 0, n2, d_T5, 0, 0, n2, d_M7, n2);
    
    // 組合結果直接寫回 C
    int c00_r = C_row_offset, c00_c = C_col_offset;
    int c01_r = C_row_offset, c01_c = C_col_offset + n2;
    int c10_r = C_row_offset + n2, c10_c = C_col_offset;
    int c11_r = C_row_offset + n2, c11_c = C_col_offset + n2;
    
    fused_strassen_combine_kernel<<<blocks, threads, 0, stream>>>(
        d_M1, d_M2, d_M3, d_M4, d_M5, d_M6, d_M7,
        d_C, C_stride, c00_r, c00_c, c01_r, c01_c, c10_r, c10_c, c11_r, c11_c, n2
    );
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

    printf("Loading data for N=%d with TILE=%d (Direct Memory Access)...\n", N, TILE);
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
    int n2 = TILE / 2;
    size_t quad_bytes = (size_t)n2 * n2 * sizeof(double);

    double *d_A, *d_B, *d_C_global;
    cudaMalloc(&d_A, (size_t)N * N * sizeof(double));
    cudaMalloc(&d_B, (size_t)N * N * sizeof(double));
    cudaMalloc(&d_C_global, (size_t)N * N * sizeof(double));

    // 每個 stream 只需要工作區（不需要 tile buffers）
    double *d_work_buffers[NUM_STREAMS][17];  // 減少到 17 個 buffers
    
    for(int s = 0; s < NUM_STREAMS; s++){
        for(int i = 0; i < 17; i++){
            cudaMalloc(&d_work_buffers[s][i], quad_bytes);
        }
    }

    cudaMemcpy(d_A, A, (size_t)N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (size_t)N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_C_global, 0, (size_t)N * N * sizeof(double));

    cudaStream_t streams[NUM_STREAMS];
    for(int s = 0; s < NUM_STREAMS; s++){
        cudaStreamCreate(&streams[s]);
    }

    printf("Computing tiled Strassen with Direct Memory Access and %d streams...\n", NUM_STREAMS);

    auto chrono_start = std::chrono::steady_clock::now();
    
    int stream_idx = 0;
    for(int ti = 0; ti < tiles; ++ti){
        for(int tj = 0; tj < tiles; ++tj){
            int s = stream_idx % NUM_STREAMS;
            cudaStream_t stream = streams[s];
            
            for(int tk = 0; tk < tiles; ++tk){
                int A_row_offset = ti * TILE;
                int A_col_offset = tk * TILE;
                int B_row_offset = tk * TILE;
                int B_col_offset = tj * TILE;
                int C_row_offset = ti * TILE;
                int C_col_offset = tj * TILE;
                
                // 直接操作大矩陣，無需拷貝 tile
                strassen_tile_multiply_direct(
                    d_A, A_row_offset, A_col_offset, N,
                    d_B, B_row_offset, B_col_offset, N,
                    d_C_global, C_row_offset, C_col_offset, N,
                    TILE, d_work_buffers[s], stream
                );
            }
            
            stream_idx++;
        }
    }

    for(int s = 0; s < NUM_STREAMS; s++){
        cudaStreamSynchronize(streams[s]);
    }
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }

    auto chrono_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = chrono_end - chrono_start;
    printf("Total computing time: %.6f s\n", elapsed.count());

    cudaMemcpy(C_result, d_C_global, (size_t)N * N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Computation complete. Checking correctness...\n");
    correctness_check(C_true, C_result, N);

    for(int s = 0; s < NUM_STREAMS; s++){
        cudaStreamDestroy(streams[s]);
        for(int i = 0; i < 17; i++){
            cudaFree(d_work_buffers[s][i]);
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_global);
    
    cleanup_memory();
    return 0;
}