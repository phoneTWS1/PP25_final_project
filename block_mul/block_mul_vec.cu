#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

int N;
float *A, *B;
float *C;
#define Bs 32

void create_matrix();
void correctness_check();
void show(float*, int);


__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
// every warp is responsible for 4 rows of a C tile
// every thread is responsible for 4 rows (same column)
// grid = dim3(n,n), block = dim3(32,8)
__global__ void block_mul_kernel(
    int N,
    int n,
    float *d_A,
    float *d_B,
    float *d_C
){
    extern __shared__ float share[];
    float *A_block = share;
    float *B_block = share + Bs * Bs;

    // block index 
    int bx = blockIdx.x;          // tile row index for A / C
    int by = blockIdx.y;          // tile col index for B / C
    int A_start_row = bx * Bs;
    int A_start_col;
    int B_start_row;
    int B_start_col = by * Bs;
    
    // block local index
    int lx = threadIdx.y;         // 0..7
    int ly = threadIdx.x;         // 0..31
    int wrap_row = lx * 4;        // this warp’s first row in the tile

    float c0=0.f, c1=0.f, c2=0.f, c3=0.f;

    // compute c
    for (int bk = 0; bk < n; bk++) {
        A_start_col = bk * Bs;
        B_start_row = bk * Bs;

        if (ly < Bs / 4) {  // 32/4 = 8 -> lanes 0..7
            int col_vec = ly;        // float4 index
            int col0    = 4 * col_vec;  // starting column in the tile

            // 4 rows per warp
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int aRow = A_start_row + wrap_row + r;
                int bRow = B_start_row + wrap_row + r;

                // base pointers to this row's tile start
                const float4 *srcA = reinterpret_cast<const float4*>(
                    &d_A[aRow * N + A_start_col]
                );
                const float4 *srcB = reinterpret_cast<const float4*>(
                    &d_B[bRow * N + B_start_col]
                );

                float4 vA = srcA[col_vec];   // loads cols [col0..col0+3]
                float4 vB = srcB[col_vec];

                int row_in_tile = wrap_row + r;

                // scatter into shared A_block
                A_block[row_in_tile * Bs + col0 + 0] = vA.x;
                A_block[row_in_tile * Bs + col0 + 1] = vA.y;
                A_block[row_in_tile * Bs + col0 + 2] = vA.z;
                A_block[row_in_tile * Bs + col0 + 3] = vA.w;

                // scatter into shared B_block
                B_block[row_in_tile * Bs + col0 + 0] = vB.x;
                B_block[row_in_tile * Bs + col0 + 1] = vB.y;
                B_block[row_in_tile * Bs + col0 + 2] = vB.z;
                B_block[row_in_tile * Bs + col0 + 3] = vB.w;
            }
        }

        __syncthreads();
        
        #pragma unroll 32
        for (int k = 0; k < Bs; k++) {
            float b  = B_block[k * Bs + ly];
            float a0 = A_block[(wrap_row + 0) * Bs + k];
            float a1 = A_block[(wrap_row + 1) * Bs + k];
            float a2 = A_block[(wrap_row + 2) * Bs + k];
            float a3 = A_block[(wrap_row + 3) * Bs + k];

            c0 += a0 * b;
            c1 += a1 * b;
            c2 += a2 * b;
            c3 += a3 * b;
        }

        __syncthreads();
    }

    // write back
    d_C[(A_start_row + wrap_row + 0) * N + B_start_col + ly] = c0;
    d_C[(A_start_row + wrap_row + 1) * N + B_start_col + ly] = c1;
    d_C[(A_start_row + wrap_row + 2) * N + B_start_col + ly] = c2;
    d_C[(A_start_row + wrap_row + 3) * N + B_start_col + ly] = c3;
}

int main(int argc, char* argv[]){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    assert(argc==2);
    N = atoi(argv[1]);
    create_matrix();

    //int Bs = 32; // Block size
    int n = N / Bs;
    assert(N % Bs ==0);

    // cudaMalloc
    float *d_A, *d_B, *d_C;
    size_t gmem = N * N *  sizeof(float);
    assert(gmem * 3 < prop.totalGlobalMem);
    cudaMalloc((void **)&d_A, gmem);
    cudaMalloc((void **)&d_B, gmem);
    cudaMalloc((void **)&d_C, gmem);
    
    // cuda Memory copy
    cudaMemcpy(d_A, A, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, gmem, cudaMemcpyHostToDevice);


    //launch kernel
    size_t shmem = Bs * Bs * 2 * sizeof(float);
    block_mul_kernel<<<dim3(n,n), dim3(Bs,8), shmem>>>(
        N,
        n,
        d_A,
        d_B,
        d_C
    );
    
    //cuda Memory copy
    cudaMemcpy(C, d_C, gmem, cudaMemcpyDeviceToHost); 
    //show(C,N);

    //free cuda memeory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

   correctness_check();

    
    return 0;
}

void create_matrix(){
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N  * sizeof(float));
    C = (float *)malloc(N * N * sizeof(float));
    for(int i = 0 ; i < N * N; i++){
        A[i] = (float)i;
        B[i] = 0.0f;
        C[i] = 0.0f;
    }

    for(int i=0 ; i<N ; i++){
        B[i * N + i] = 1.0f;
    }
    
}

void correctness_check(){
    int cnt = 0;
    for(int i = 0 ; i < N * N ; i++){
        if(C[i] != A[i])
            break;
        cnt++;
    }
    if(cnt != N * N){
        printf("failed: cnt = %d\n ",cnt);
    }else{
        printf("success\n");
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