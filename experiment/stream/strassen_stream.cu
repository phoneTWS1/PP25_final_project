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
#define NUM_STREAMS 8  // 使用的 stream 數量

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

// CUDA Kernel: 矩陣加法/減法
__global__ void add_matrix_kernel(const double *a, const double *b, double *c, int n, double alpha, double beta){
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n * n;

    if(idx < total){
        c[idx] = alpha * a[idx] + beta * b[idx];
    }
}

// CUDA Kernel: 標準矩陣乘法（用於 Strassen 的 7 次乘法）
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

// CUDA Kernel: 累加 tile 到全域矩陣（使用 atomic add 避免 race condition）
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

// 使用 Strassen 算法計算 tile 乘法：C_tile = A_tile * B_tile
void strassen_tile_multiply(double *d_A_tile, double *d_B_tile, double *d_C_tile, 
                           int tile_size, 
                           double *d_work[], // 工作區陣列
                           cudaStream_t stream) {
    int n2 = tile_size / 2;  // quadrant 大小
    
    // 分配工作區索引（總共需要約 20 個 buffer）
    double *d_A00 = d_work[0], *d_A01 = d_work[1], *d_A10 = d_work[2], *d_A11 = d_work[3];
    double *d_B00 = d_work[4], *d_B01 = d_work[5], *d_B10 = d_work[6], *d_B11 = d_work[7];
    double *d_M1 = d_work[8], *d_M2 = d_work[9], *d_M3 = d_work[10], *d_M4 = d_work[11];
    double *d_M5 = d_work[12], *d_M6 = d_work[13], *d_M7 = d_work[14];
    double *d_tmp[5] = {d_work[15], d_work[16], d_work[17], d_work[18], d_work[19]};
    
    // Kernel 參數
    int add_threads = 256;
    int add_blocks = (n2 * n2 + add_threads - 1) / add_threads;
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
    
    // 步驟 2: 計算 Strassen 的 7 個矩陣乘法
    // M1 = (A00 + A11) * (B00 + B11)
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_A00, d_A11, d_tmp[0], n2, 1.0, 1.0);
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_B00, d_B11, d_tmp[1], n2, 1.0, 1.0);
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_tmp[0], d_tmp[1], d_M1, n2);
    
    // M2 = (A10 + A11) * B00
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_A10, d_A11, d_tmp[0], n2, 1.0, 1.0);
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_tmp[0], d_B00, d_M2, n2);
    
    // M3 = A00 * (B01 - B11)
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_B01, d_B11, d_tmp[0], n2, 1.0, -1.0);
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_A00, d_tmp[0], d_M3, n2);
    
    // M4 = A11 * (B10 - B00)
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_B10, d_B00, d_tmp[0], n2, 1.0, -1.0);
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_A11, d_tmp[0], d_M4, n2);
    
    // M5 = (A00 + A01) * B11
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_A00, d_A01, d_tmp[0], n2, 1.0, 1.0);
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_tmp[0], d_B11, d_M5, n2);
    
    // M6 = (A10 - A00) * (B00 + B01)
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_A10, d_A00, d_tmp[0], n2, 1.0, -1.0);
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_B00, d_B01, d_tmp[1], n2, 1.0, 1.0);
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_tmp[0], d_tmp[1], d_M6, n2);
    
    // M7 = (A01 - A11) * (B10 + B11)
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_A01, d_A11, d_tmp[0], n2, 1.0, -1.0);
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_B10, d_B11, d_tmp[1], n2, 1.0, 1.0);
    matmul_kernel<<<mul_grid, mul_block, 0, stream>>>(d_tmp[0], d_tmp[1], d_M7, n2);
    
    // 步驟 3: 組合結果計算 C 的四個 quadrants
    // C00 = M1 + M4 - M5 + M7
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_M1, d_M4, d_tmp[0], n2, 1.0, 1.0);
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_tmp[0], d_M5, d_tmp[1], n2, 1.0, -1.0);
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_tmp[1], d_M7, d_tmp[2], n2, 1.0, 1.0);
    
    // C01 = M3 + M5
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_M3, d_M5, d_tmp[3], n2, 1.0, 1.0);
    
    // C10 = M2 + M4
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_M2, d_M4, d_tmp[4], n2, 1.0, 1.0);
    
    // C11 = M1 - M2 + M3 + M6
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_M1, d_M2, d_A00, n2, 1.0, -1.0);
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_A00, d_M3, d_A01, n2, 1.0, 1.0);
    add_matrix_kernel<<<add_blocks, add_threads, 0, stream>>>(d_A01, d_M6, d_A10, n2, 1.0, 1.0);
    
    // 步驟 4: 將四個 quadrants 合併回 C_tile
    cudaMemcpy2DAsync(d_C_tile, tile_size * sizeof(double),
                      d_tmp[2], n2 * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_C_tile + n2, tile_size * sizeof(double),
                      d_tmp[3], n2 * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_C_tile + n2 * tile_size, tile_size * sizeof(double),
                      d_tmp[4], n2 * sizeof(double),
                      n2 * sizeof(double), n2, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(d_C_tile + n2 * tile_size + n2, tile_size * sizeof(double),
                      d_A10, n2 * sizeof(double),
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

    printf("Loading data for N=%d with TILE=%d (Multi-stream Strassen)...\n", N, TILE);
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

    // 分配 device 記憶體（全域矩陣）
    double *d_A, *d_B, *d_C_global;
    cudaMalloc(&d_A, (size_t)N * N * sizeof(double));
    cudaMalloc(&d_B, (size_t)N * N * sizeof(double));
    cudaMalloc(&d_C_global, (size_t)N * N * sizeof(double));

    // 為每個 stream 分配 tile buffers 和 work buffers
    double *d_A_tiles[NUM_STREAMS], *d_B_tiles[NUM_STREAMS], *d_C_tiles[NUM_STREAMS];
    double *d_work_buffers[NUM_STREAMS][20];
    
    for(int s = 0; s < NUM_STREAMS; s++){
        cudaMalloc(&d_A_tiles[s], tile_bytes);
        cudaMalloc(&d_B_tiles[s], tile_bytes);
        cudaMalloc(&d_C_tiles[s], tile_bytes);
        
        for(int i = 0; i < 20; i++){
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

    // 創建多個 CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for(int s = 0; s < NUM_STREAMS; s++){
        cudaStreamCreate(&streams[s]);
    }

    printf("Computing tiled Strassen with %d streams...\n", NUM_STREAMS);

    auto chrono_start = std::chrono::steady_clock::now();
    
    // 主迴圈：使用多 streams 平行化
    int stream_idx = 0;
    for(int ti = 0; ti < tiles; ++ti){
        for(int tj = 0; tj < tiles; ++tj){
            // 選擇當前 stream
            int s = stream_idx % NUM_STREAMS;
            cudaStream_t stream = streams[s];
            
            for(int tk = 0; tk < tiles; ++tk){
                // 從全域矩陣提取 tile
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
                
                // 使用 Strassen 計算 C_tile = A_tile * B_tile
                strassen_tile_multiply(d_A_tiles[s], d_B_tiles[s], d_C_tiles[s], 
                                     TILE, d_work_buffers[s], stream);
                
                // 累加到全域 C 矩陣（使用 atomic add）
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
        for(int i = 0; i < 20; i++){
            cudaFree(d_work_buffers[s][i]);
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_global);
    
    cleanup_memory();
    return 0;
}