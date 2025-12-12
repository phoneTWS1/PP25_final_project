#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <chrono>

int N;
float *A, *B, *C_true;
float *C;
#define Bs 32
int TILE = 512;

void load_matrix(float**, int, const char *filename);
void correctness_check(const float*, const float*, int);
void show(float*, int);

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}



// 修改：支援可變 coarse TILE（must be multiple of Bs）
// blockDim = dim3(Bs, 8) (固定)
// 每個 block 負責一個 TILE x TILE 的輸出區塊，內部以 Bs x Bs 的 shared-loading 反覆累加
// 每個 thread 原本負責 4 rows in a Bs-subtile（wrap of 4 rows），在 TILE > Bs 時會處理多個 subtiles
__global__ void block_mul_kernel(
    int N,
    int TILE,            // coarse tile size (multiple of Bs)
    float *d_A,
    float *d_B,
    float *d_C
){
    extern __shared__ float share[];
    float *A_block = share;                         // Bs * Bs
    float *B_block = share + Bs * Bs;              // Bs * Bs

    // block index -> tile coordinates
    int tile_col = blockIdx.x; // x -> columns of tiles
    int tile_row = blockIdx.y; // y -> rows of tiles

    int tile_row_base = tile_row * TILE;
    int tile_col_base = tile_col * TILE;

    // local thread indices
    int lx = threadIdx.y; // 0..7
    int ly = threadIdx.x; // 0..31

    // threads per warp-group handle 4 rows per Bs-subtile (same as original)
    int wrap_row_base = lx * 4; // within a Bs-subtile: 0,4,8,...,28

    int R = TILE / Bs; // number of Bs-subtiles per TILE (assume TILE % Bs == 0)

    // For each sub-output sub-tile inside this coarse TILE
    for (int ii = 0; ii < R; ++ii) {
        for (int jj = 0; jj < R; ++jj) {
            // compute the subtile's top-left coords
            int A_sub_row = tile_row_base + ii * Bs; // row base for A subtile
            int B_sub_col = tile_col_base + jj * Bs; // col base for B subtile

            // accumulators for 4 rows handled by this thread in the Bs-subtile
            float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

            // iterate over k-blocks (Bs-sized) across the full matrix
            int num_kblocks = N / Bs;
            for (int bk = 0; bk < num_kblocks; ++bk) {
                int A_sub_col = bk * Bs;
                int B_sub_row = bk * Bs;

                // load A_block (each thread writes 4 elements in column 'ly')
                // bounds check safe because we require N % Bs == 0
                A_block[(wrap_row_base + 0) * Bs + ly] = d_A[(A_sub_row + wrap_row_base + 0) * N + A_sub_col + ly];
                A_block[(wrap_row_base + 1) * Bs + ly] = d_A[(A_sub_row + wrap_row_base + 1) * N + A_sub_col + ly];
                A_block[(wrap_row_base + 2) * Bs + ly] = d_A[(A_sub_row + wrap_row_base + 2) * N + A_sub_col + ly];
                A_block[(wrap_row_base + 3) * Bs + ly] = d_A[(A_sub_row + wrap_row_base + 3) * N + A_sub_col + ly];

                // load B_block (Bs x Bs) for this (bk, jj) position
                B_block[(wrap_row_base + 0) * Bs + ly] = d_B[(B_sub_row + wrap_row_base + 0) * N + B_sub_col + ly];
                B_block[(wrap_row_base + 1) * Bs + ly] = d_B[(B_sub_row + wrap_row_base + 1) * N + B_sub_col + ly];
                B_block[(wrap_row_base + 2) * Bs + ly] = d_B[(B_sub_row + wrap_row_base + 2) * N + B_sub_col + ly];
                B_block[(wrap_row_base + 3) * Bs + ly] = d_B[(B_sub_row + wrap_row_base + 3) * N + B_sub_col + ly];

                __syncthreads();

                // compute inner product for the 4 rows handled by this thread
                #pragma unroll 32
                for (int k = 0; k < Bs; ++k) {
                    float b = B_block[k * Bs + ly];
                    float a0 = A_block[(wrap_row_base + 0) * Bs + k];
                    float a1 = A_block[(wrap_row_base + 1) * Bs + k];
                    float a2 = A_block[(wrap_row_base + 2) * Bs + k];
                    float a3 = A_block[(wrap_row_base + 3) * Bs + k];

                    c0 += a0 * b;
                    c1 += a1 * b;
                    c2 += a2 * b;
                    c3 += a3 * b;
                }

                __syncthreads();
            } // bk

            // write back results for this subtile (4 rows)
            d_C[(A_sub_row + wrap_row_base + 0) * N + B_sub_col + ly] += c0;
            d_C[(A_sub_row + wrap_row_base + 1) * N + B_sub_col + ly] += c1;
            d_C[(A_sub_row + wrap_row_base + 2) * N + B_sub_col + ly] += c2;
            d_C[(A_sub_row + wrap_row_base + 3) * N + B_sub_col + ly] += c3;
        } // jj
    } // ii
}

int main(int argc, char* argv[]){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (!(argc == 5 || argc == 6)) {
        fprintf(stderr, "usage: %s N A_file B_file C_true_file [TILE]\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    const char *a_filename = argv[2];
    const char *b_filename = argv[3];
    const char *c_true_filename = argv[4];
    if (argc == 6) TILE = atoi(argv[5]);

    if (N % Bs != 0) { fprintf(stderr, "Error: N must be divisible by %d\n", Bs); return 1; }
    if (TILE % Bs != 0) { fprintf(stderr, "Error: TILE must be multiple of %d\n", Bs); return 1; }
    if (N % TILE != 0) { fprintf(stderr, "Error: N must be divisible by TILE\n"); return 1; }

    load_matrix(&A, N, a_filename);
    load_matrix(&B, N, b_filename);
    load_matrix(&C_true, N, c_true_filename);
    C = (float*)calloc((size_t)N * N, sizeof(float));
    assert(C);

    int tiles = N / TILE;

    // cudaMalloc
    float *d_A, *d_B, *d_C;
    size_t gmem = (size_t)N * N *  sizeof(float);
    assert(gmem * 3 < prop.totalGlobalMem);
    cudaMalloc((void **)&d_A, gmem);
    cudaMalloc((void **)&d_B, gmem);
    cudaMalloc((void **)&d_C, gmem);
    
    // cuda Memory copy
    cudaMemcpy(d_A, A, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, gmem, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, gmem);

    //launch kernel
    auto chrono_start = std::chrono::steady_clock::now();
    size_t shmem = Bs * Bs * 2 * sizeof(float); // shared mem holds two Bs x Bs tiles
    dim3 grid(tiles, tiles);
    dim3 block(Bs, 8);
    block_mul_kernel<<<grid, block, shmem>>>(
        N,
        TILE,
        d_A,
        d_B,
        d_C
    );

    auto chrono_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = chrono_end - chrono_start;
    printf("Total computing time: %.6f s\n", elapsed.count());

    //cuda Memory copy
    cudaMemcpy(C, d_C, gmem, cudaMemcpyDeviceToHost); 

    //free cuda memeory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    correctness_check(C_true, C, N);

    free(C);
    return 0;
}

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

void correctness_check(const float *C_true, const float *C_result, int N){
    int mismatch_count = 0;
    float tol = 5e-3f * double(N) * double(N);
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

void show(float* M, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N ;j++){
            printf("%.0f, ", M[i * N+ j]);
        }
        printf("\n");
    }
    printf("\n");
}