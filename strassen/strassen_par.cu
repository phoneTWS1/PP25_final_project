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
// Kernel Fusion 1: 合併矩陣加法和乘法的準備階段
// 一次 kernel 完成多個加減法操作
// ============================================================================
__global__ void fused_strassen_prep_kernel(
    const double *A00, const double *A01, const double *A10, const double *A11,
    const double *B00, const double *B01, const double *B10, const double *B11,
    double *S1, double *S2, double *S3, double *S4,  // A 的中間結果
    double *T1, double *T2, double *T3, double *T4,  // B 的中間結果
    int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    if(idx < total) {
        // S1 = A00 + A11  (for M1)
        S1[idx] = A00[idx] + A11[idx];
        
        // S2 = A10 + A11  (for M2)
        S2[idx] = A10[idx] + A11[idx];
        
        // S3 = A00 + A01  (for M5)
        S3[idx] = A00[idx] + A01[idx];
        
        // S4 = A10 - A00  (for M6)
        S4[idx] = A10[idx] - A00[idx];
        
        // T1 = B00 + B11  (for M1)
        T1[idx] = B00[idx] + B11[idx];
        
        // T2 = B01 - B11  (for M3)
        T2[idx] = B01[idx] - B11[idx];
        
        // T3 = B10 - B00  (for M4)
        T3[idx] = B10[idx] - B00[idx];
        
        // T4 = B00 + B01  (for M6)
        T4[idx] = B00[idx] + B01[idx];
    }
}

// ============================================================================
// Kernel Fusion 2: 合併最後的結果組合
// 一次 kernel 完成所有四個 quadrant 的計算
// ============================================================================
__global__ void fused_strassen_combine_kernel(
    const double *M1, const double *M2, const double *M3, 
    const double *M4, const double *M5, const double *M6, const double *M7,
    double *C00, double *C01, double *C10, double *C11,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    if(idx < total) {
        // C00 = M1 + M4 - M5 + M7
        C00[idx] = M1[idx] + M4[idx] - M5[idx] + M7[idx];
        
        // C01 = M3 + M5
        C01[idx] = M3[idx] + M5[idx];
        
        // C10 = M2 + M4
        C10[idx] = M2[idx] + M4[idx];
        
        // C11 = M1 - M2 + M3 + M6
        C11[idx] = M1[idx] - M2[idx] + M3[idx] + M6[idx];
    }
}

// ============================================================================
// Kernel Fusion 3: 合併 M7 的準備和其他操作
// ============================================================================
__global__ void fused_m7_prep_kernel(
    const double *A01, const double *A11,
    const double *B10, const double *B11,
    double *S5, double *T5,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    if(idx < total) {
        // S5 = A01 - A11  (for M7)
        S5[idx] = A01[idx] - A11[idx];
        
        // T5 = B10 + B11  (for M7)
        T5[idx] = B10[idx] + B11[idx];
    }
}

// ============================================================================
// 標準矩陣乘法 kernel (保持不變)
// ============================================================================
__global__ void matmul_kernel(const double *a, const double *b, double *c, int n){
    __shared__ double As[32][32];
    __shared__ double Bs[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;

    double sum = 0.0;
    int numTiles = (n + 31) / 32;

    for(int t = 0; t < numTiles; t++){
        int aRow = row;
        int aCol = t * 32 + tx;
        if(aRow < n && aCol < n){
            As[ty][tx] = a[aRow * n + aCol];
        } else {
            As[ty][tx] = 0.0;
        }

        int bRow = t * 32 + ty;
        int bCol = col;
        if(bRow < n && bCol < n){
            Bs[ty][tx] = b[bRow * n + bCol];
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
// 累加 kernel (保持不變)
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
// 優化後的 Strassen tile multiply 函數
// ============================================================================
void strassen_tile_multiply_fused(
    double *d_A_tile, double *d_B_tile, double *d_C_tile, 
    int tile_size, 
    double *d_work[],
    cudaStream_t stream) 
{
    int n2 = tile_size / 2;
    
    // 分配工作區索引
    double *d_A00 = d_work[0], *d_A01 = d_work[1], *d_A10 = d_work[2], *d_A11 = d_work[3];
    double *d_B00 = d_work[4], *d_B01 = d_work[5], *d_B10 = d_work[6], *d_B11 = d_work[7];
    double *d_M1 = d_work[8], *d_M2 = d_work[9], *d_M3 = d_work[10], *d_M4 = d_work[11];
    double *d_M5 = d_work[12], *d_M6 = d_work[13], *d_M7 = d_work[14];
    
    // 用於 fused kernels 的中間結果
    double *d_S1 = d_work[15], *d_S2 = d_work[16], *d_S3 = d_work[17];
    double *d_S4 = d_work[18], *d_S5 = d_work[19];
    double *d_T1 = d_work[20], *d_T2 = d_work[21], *d_T3 = d_work[22];
    double *d_T4 = d_work[23], *d_T5 = d_work[24];
    
    double *d_C00 = d_work[25], *d_C01 = d_work[26], *d_C10 = d_work[27], *d_C11 = d_work[28];
    
    // Kernel 參數
    int threads = 256;
    int blocks = (n2 * n2 + threads - 1) / threads;
    dim3 mul_block(32, 32);
    dim3 mul_grid((n2 + 31) / 32, (n2 + 31) / 32);
    
    // 步驟 1: 分割 A 和 B 成 4 個 quadrants
    cudaMemcpy2DAsync(d_A00, n2 * sizeof(double), 
                      d_A_tile, tile_size * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_A01, n2 * sizeof(double), 
                      d_A_tile + n2, tile_size * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_A10, n2 * sizeof(double), 
                      d_A_tile + n2 * tile_size, tile_size * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_A11, n2 * sizeof(double), 
                      d_A_tile + n2 * tile_size + n2, tile_size * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    
    cudaMemcpy2DAsync(d_B00, n2 * sizeof(double), 
                      d_B_tile, tile_size * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_B01, n2 * sizeof(double), 
                      d_B_tile + n2, tile_size * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_B10, n2 * sizeof(double), 
                      d_B_tile + n2 * tile_size, tile_size * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_B11, n2 * sizeof(double), 
                      d_B_tile + n2 * tile_size + n2, tile_size * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    
    // 步驟 2: 使用 fused kernel 準備 M1-M6 的輸入 (一次完成 8 個加減法)
    fused_strassen_prep_kernel<<<blocks, threads, 0, stream>>>(
        d_A00, d_A01, d_A10, d_A11,
        d_B00, d_B01, d_B10, d_B11,
        d_S1, d_S2, d_S3, d_S4,
        d_T1, d_T2, d_T3, d_T4,
        n2
    );
    
    // 步驟 3: 計算 M1-M6
    // M1 = S1 * T1 = (A00 + A11) * (B00 + B11)
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_S1, d_T1, d_M1, n2);
    
    // M2 = S2 * B00 = (A10 + A11) * B00
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_S2, d_B00, d_M2, n2);
    
    // M3 = A00 * T2 = A00 * (B01 - B11)
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_A00, d_T2, d_M3, n2);
    
    // M4 = A11 * T3 = A11 * (B10 - B00)
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_A11, d_T3, d_M4, n2);
    
    // M5 = S3 * B11 = (A00 + A01) * B11
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_S3, d_B11, d_M5, n2);
    
    // M6 = S4 * T4 = (A10 - A00) * (B00 + B01)
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_S4, d_T4, d_M6, n2);
    
    // 步驟 4: 準備 M7 的輸入 (單獨的 fused kernel)
    fused_m7_prep_kernel<<<blocks, threads, 0, stream>>>(
        d_A01, d_A11, d_B10, d_B11,
        d_S5, d_T5, n2
    );
    
    // M7 = S5 * T5 = (A01 - A11) * (B10 + B11)
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_S5, d_T5, d_M7, n2);
    
    // 步驟 5: 使用 fused kernel 組合所有結果 (一次完成 4 個 quadrant)
    fused_strassen_combine_kernel<<<blocks, threads, 0, stream>>>(
        d_M1, d_M2, d_M3, d_M4, d_M5, d_M6, d_M7,
        d_C00, d_C01, d_C10, d_C11,
        n2
    );
    
    // 步驟 6: 將四個 quadrants 合併回 C_tile
    cudaMemcpy2DAsync(d_C_tile, tile_size * sizeof(double),
                      d_C00, n2 * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_C_tile + n2, tile_size * sizeof(double),
                      d_C01, n2 * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_C_tile + n2 * tile_size, tile_size * sizeof(double),
                      d_C10, n2 * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_C_tile + n2 * tile_size + n2, tile_size * sizeof(double),
                      d_C11, n2 * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
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

    printf("Loading data for N=%d with TILE=%d (Kernel Fusion Strassen)...\n", N, TILE);
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
    cudaMalloc(&d_A, (size_t)N * N * sizeof(double));
    cudaMalloc(&d_B, (size_t)N * N * sizeof(double));
    cudaMalloc(&d_C_global, (size_t)N * N * sizeof(double));

    // 為每個 stream 分配 buffers (需要更多 work buffers 給 fused kernels)
    double *d_A_tiles[NUM_STREAMS], *d_B_tiles[NUM_STREAMS], *d_C_tiles[NUM_STREAMS];
    double *d_work_buffers[NUM_STREAMS][29];  // 增加到 29 個 buffers
    
    for(int s = 0; s < NUM_STREAMS; s++){
        cudaMalloc(&d_A_tiles[s], tile_bytes);
        cudaMalloc(&d_B_tiles[s], tile_bytes);
        cudaMalloc(&d_C_tiles[s], tile_bytes);
        
        for(int i = 0; i < 29; i++){
            cudaMalloc(&d_work_buffers[s][i], quad_bytes);
        }
    }

    // 拷貝輸入到 device
    cudaMemcpy(d_A, A, (size_t)N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (size_t)N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_C_global, 0, (size_t)N * N * sizeof(double));

    // 累加 kernel 參數
    int acc_threads = 1024;
    int acc_blocks = (TILE * TILE + acc_threads - 1) / acc_threads;

    // 創建 CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for(int s = 0; s < NUM_STREAMS; s++){
        cudaStreamCreate(&streams[s]);
    }

    printf("Computing tiled Strassen with Kernel Fusion and %d streams...\n", NUM_STREAMS);

    auto chrono_start = std::chrono::steady_clock::now();
    
    // 主迴圈
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
                
                cudaMemcpy2DAsync(d_A_tiles[s], TILE * sizeof(double),
                           d_A + A_row_offset * N + A_col_offset, N * sizeof(double),
                           TILE * sizeof(double), TILE,
                           cudaMemcpyDeviceToDevice, stream);
                
                cudaMemcpy2DAsync(d_B_tiles[s], TILE * sizeof(double),
                           d_B + B_row_offset * N + B_col_offset, N * sizeof(double),
                           TILE * sizeof(double), TILE,
                           cudaMemcpyDeviceToDevice, stream);
                
                // 使用 kernel fusion 版本的 Strassen
                strassen_tile_multiply_fused(d_A_tiles[s], d_B_tiles[s], d_C_tiles[s], 
                                           TILE, d_work_buffers[s], stream);
                
                int C_row_offset = ti * TILE;
                int C_col_offset = tj * TILE;
                accumulate_kernel<<<acc_blocks, acc_threads, 0, stream>>>(
                    d_C_global, d_C_tiles[s], C_row_offset, C_col_offset, TILE, N);
            }
            
            stream_idx++;
        }
    }

    // 同步所有 streams
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

    // 清理資源
    for(int s = 0; s < NUM_STREAMS; s++){
        cudaStreamDestroy(streams[s]);
        cudaFree(d_A_tiles[s]);
        cudaFree(d_B_tiles[s]);
        cudaFree(d_C_tiles[s]);
        for(int i = 0; i < 29; i++){
            cudaFree(d_work_buffers[s][i]);
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_global);
    
    cleanup_memory();
    return 0;
}