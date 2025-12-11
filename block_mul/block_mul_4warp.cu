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
// every wrap is reposible for 8 row of a C tile
// every thread is reposible for 8 rows elements (same column)
// grid = dim3(n,n), block = dim3(32,4)
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
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int A_start_row = bx * Bs;
    int A_start_col;
    int B_start_row;
    int B_start_col = by * Bs;
    
    // block local index
    int lx = threadIdx.y; // threadIdx.y = 0...7, 
    int ly = threadIdx.x; // threadIdx.x = 0...31
    int warp_row = lx * 8;


    float c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0;
    float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;

   // compute c
    for(int bk = 0; bk< n ; bk++){
        A_start_col = bk * Bs;
        B_start_row = bk * Bs;
        
        // load A, B block
        A_block[warp_row * Bs + ly] = d_A[(A_start_row + warp_row) * N + A_start_col + ly];
        A_block[(warp_row + 1) * Bs + ly] = d_A[(A_start_row + warp_row + 1) * N + A_start_col + ly];
        A_block[(warp_row + 2) * Bs + ly] = d_A[(A_start_row + warp_row + 2) * N + A_start_col + ly];
        A_block[(warp_row + 3) * Bs + ly] = d_A[(A_start_row + warp_row + 3) * N + A_start_col + ly];
        A_block[(warp_row + 4) * Bs + ly] = d_A[(A_start_row + warp_row + 4) * N + A_start_col + ly];
        A_block[(warp_row + 5) * Bs + ly] = d_A[(A_start_row + warp_row + 5) * N + A_start_col + ly];
        A_block[(warp_row + 6) * Bs + ly] = d_A[(A_start_row + warp_row + 6) * N + A_start_col + ly];
        A_block[(warp_row + 7) * Bs + ly] = d_A[(A_start_row + warp_row + 7) * N + A_start_col + ly];

        B_block[warp_row * Bs + ly] = d_B[(B_start_row + warp_row) * N + B_start_col + ly];
        B_block[(warp_row + 1) * Bs + ly] = d_B[(B_start_row + warp_row + 1) * N + B_start_col + ly];
        B_block[(warp_row + 2) * Bs + ly] = d_B[(B_start_row + warp_row + 2) * N + B_start_col + ly];
        B_block[(warp_row + 3) * Bs + ly] = d_B[(B_start_row + warp_row + 3) * N + B_start_col + ly];
        B_block[(warp_row + 4) * Bs + ly] = d_B[(B_start_row + warp_row + 4) * N + B_start_col + ly];
        B_block[(warp_row + 5) * Bs + ly] = d_B[(B_start_row + warp_row + 5) * N + B_start_col + ly];
        B_block[(warp_row + 6) * Bs + ly] = d_B[(B_start_row + warp_row + 6) * N + B_start_col + ly];
        B_block[(warp_row + 7) * Bs + ly] = d_B[(B_start_row + warp_row + 7) * N + B_start_col + ly];

    
        __syncthreads();
        
        #pragma unroll 32
        for(int k = 0 ; k < Bs ; k ++){
            float b = B_block[k * Bs + ly];
            // 8 FMAs 9 load => 0.888 FMAs per load
            a0 = A_block[(warp_row * Bs) + k];
            a1 = A_block[(warp_row + 1) * Bs + k];
            a2 = A_block[(warp_row + 2) * Bs + k];
            a3 = A_block[(warp_row + 3) * Bs + k];
            a4 = A_block[(warp_row + 4) * Bs + k];
            a5 = A_block[(warp_row + 5) * Bs + k];
            a6 = A_block[(warp_row + 6) * Bs + k];
            a7 = A_block[(warp_row + 7) * Bs + k];

            c0 += a0 * b;
            c1 += a1 * b;
            c2 += a2 * b;
            c3 += a3 * b;
            c4 += a4 * b;
            c5 += a5 * b;
            c6 += a6 * b;
            c7 += a7 * b;
            
            
        }
        __syncthreads();
    }

    //write back
    d_C[(A_start_row + warp_row) * N + B_start_col + ly] = c0;
    d_C[(A_start_row + warp_row + 1) * N + B_start_col + ly] = c1;
    d_C[(A_start_row + warp_row + 2) * N + B_start_col + ly] = c2;
    d_C[(A_start_row + warp_row + 3) * N + B_start_col + ly] = c3;
    d_C[(A_start_row + warp_row + 4) * N + B_start_col + ly] = c4;
    d_C[(A_start_row + warp_row + 5) * N + B_start_col + ly] = c5;
    d_C[(A_start_row + warp_row + 6) * N + B_start_col + ly] = c6;
    d_C[(A_start_row + warp_row + 7) * N + B_start_col + ly] = c7;

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
    block_mul_kernel<<<dim3(n,n), dim3(Bs,4), shmem>>>(
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