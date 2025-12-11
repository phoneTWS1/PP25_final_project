#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int N;
static float *A, *B, *C_result, *C_true; 
static float *A00, *A01, *A10, *A11; 
static float *B00, *B01, *B10, *B11;

void load_matrix(float **mat_ptr, int N_dim, const char *filename) {
    long long size = (long long)N_dim * N_dim;
    size_t bytes = size * sizeof(float);
    
    FILE *file = fopen(filename, "rb"); 
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    *mat_ptr = (float *)malloc(bytes);
    if (*mat_ptr == NULL) {
        perror("Host malloc failed in load_matrix");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    size_t read_count = fread(*mat_ptr, sizeof(float), size, file);
    if (read_count != size) {
        fprintf(stderr, "Error: Read incomplete from %s. Expected %lld elements, read %zu.\n", 
                filename, size, read_count);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fclose(file);
}

void divide(int n2){
    
    A00 = (float*)calloc(n2 * n2, sizeof(float)); 
    A01 = (float*)calloc(n2 * n2, sizeof(float)); 
    A10 = (float*)calloc(n2 * n2, sizeof(float)); 
    A11 = (float*)calloc(n2 * n2, sizeof(float));
    B00 = (float*)calloc(n2 * n2, sizeof(float)); 
    B01 = (float*)calloc(n2 * n2, sizeof(float)); 
    B10 = (float*)calloc(n2 * n2, sizeof(float)); 
    B11 = (float*)calloc(n2 * n2, sizeof(float));

    size_t n2row = n2 * sizeof(float);
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
    size_t n2row = n2 * sizeof(float);
    for(int i=0; i<n2; i++){
        memcpy(C_result + i*N, A00 + i*n2, n2row);
        memcpy(C_result + i*N + n2, A01 + i*n2, n2row);
        memcpy(C_result + (n2 + i)*N, A10 + i*n2, n2row);
        memcpy(C_result + (n2 + i)*N + n2, A11 + i*n2, n2row);
    }
}

// CUDA Kernel: 矩陣加法/減法
__global__ void add_matrix_kernel(const float *a, const float *b, float *c, int n, float alpha, float beta){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    if(idx < total){
        c[idx] = alpha * a[idx] + beta * b[idx];
    }
}

// CUDA Kernel: Tiled 矩陣乘法
__global__ void matmul_kernel(const float *a, const float *b, float *c, int n){
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * 32 + ty;
    int col = bx * 32 + tx;
    
    float sum = 0.0f;
    
    int numTiles = (n + 31) / 32;
    
    for(int t = 0; t < numTiles; t++){
        // 載入 A 的 tile
        int aRow = row;
        int aCol = t * 32 + tx;
        if(aRow < n && aCol < n){
            As[ty][tx] = a[aRow * n + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // 載入 B 的 tile
        int bRow = t * 32 + ty;
        int bCol = col;
        if(bRow < n && bCol < n){
            Bs[ty][tx] = b[bRow * n + bCol];
        } else {
            Bs[ty][tx] = 0.0f;
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

void correctness_check(const float *C_true, const float *C_result, int N){
    int mismatch_count = 0;
    float tol = 5e-3f;
    long long sz = (long long)N * N;
    float max_error = 0.0f;
    
    for(long long i=0; i < sz; i++){
        float error = fabsf(C_result[i] - C_true[i]);
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
    
    C_result = (float*)calloc(N * N, sizeof(float));
    if (C_result == NULL) {
        perror("Host malloc failed for C_result");
        cleanup_memory();
        return 1;
    }
    
    int n2 = N/2;
    divide(n2);

    // 分配 GPU 記憶體
    float *d_A00, *d_A01, *d_A10, *d_A11;
    float *d_B00, *d_B01, *d_B10, *d_B11;
    float *d_m1, *d_m2, *d_m3, *d_m4, *d_m5, *d_m6, *d_m7;
    float *d_C00, *d_C01, *d_C10, *d_C11;
    float *d_tmp[10];
    
    size_t bytes = n2 * n2 * sizeof(float);
    
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
    
    // 複製資料到 GPU
    cudaMemcpy(d_A00, A00, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A01, A01, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A10, A10, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A11, A11, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B00, B00, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B01, B01, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B10, B10, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B11, B11, bytes, cudaMemcpyHostToDevice);
    
    // 設定 kernel 參數
    int threadsPerBlock = 256;
    int blocksPerGrid = (n2 * n2 + threadsPerBlock - 1) / threadsPerBlock;
    
    dim3 matmul_block(32, 32);
    dim3 matmul_grid((n2 + 31) / 32, (n2 + 31) / 32);
    
    // 計時開始
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Strassen 的 10 個加減法
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A00, d_A11, d_tmp[0], n2, 1.0f, 1.0f);  // S1
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B00, d_B11, d_tmp[1], n2, 1.0f, 1.0f);  // S2
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A10, d_A11, d_tmp[2], n2, 1.0f, 1.0f);  // S3
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B01, d_B11, d_tmp[3], n2, 1.0f, -1.0f); // S4
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B10, d_B00, d_tmp[4], n2, 1.0f, -1.0f); // S5
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A00, d_A01, d_tmp[5], n2, 1.0f, 1.0f);  // S6
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A10, d_A00, d_tmp[6], n2, 1.0f, -1.0f); // S7
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B00, d_B01, d_tmp[7], n2, 1.0f, 1.0f);  // S8
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A01, d_A11, d_tmp[8], n2, 1.0f, -1.0f); // S9
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_B10, d_B11, d_tmp[9], n2, 1.0f, 1.0f);  // S10

    // 7 次矩陣乘法
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_tmp[0], d_tmp[1], d_m1, n2); // M1 = (A00+A11)*(B00+B11)
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_tmp[2], d_B00, d_m2, n2);    // M2 = (A10+A11)*B00
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_A00, d_tmp[3], d_m3, n2);    // M3 = A00*(B01-B11)
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_A11, d_tmp[4], d_m4, n2);    // M4 = A11*(B10-B00)
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_tmp[5], d_B11, d_m5, n2);    // M5 = (A00+A01)*B11
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_tmp[6], d_tmp[7], d_m6, n2); // M6 = (A10-A00)*(B00+B01)
    matmul_kernel<<<matmul_grid, matmul_block>>>(d_tmp[8], d_tmp[9], d_m7, n2); // M7 = (A01-A11)*(B10+B11)

    // 組合結果
    // C00 = M1 + M4 - M5 + M7
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m1, d_m7, d_tmp[1], n2, 1.0f, 1.0f);
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m4, d_m5, d_tmp[2], n2, 1.0f, -1.0f);
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tmp[1], d_tmp[2], d_C00, n2, 1.0f, 1.0f);
    
    // C01 = M3 + M5
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m3, d_m5, d_C01, n2, 1.0f, 1.0f);

    // C10 = M2 + M4
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m2, d_m4, d_C10, n2, 1.0f, 1.0f);

    // C11 = M1 - M2 + M3 + M6
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m1, d_m2, d_tmp[4], n2, 1.0f, -1.0f);
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m3, d_m6, d_tmp[5], n2, 1.0f, 1.0f);
    add_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tmp[4], d_tmp[5], d_C11, n2, 1.0f, 1.0f);

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
    
    // 複製結果回 Host
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