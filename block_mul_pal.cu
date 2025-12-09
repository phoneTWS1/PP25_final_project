#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

int N;
float *A, *B;
float *C;

void create_matrix(){
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N  * sizeof(float));
    C = (float *)malloc(N * N * sizeof(float));
    for(int i = 0 ; i < N * N; i++){
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }
    
}

void correctness_check(){
    int cnt = 0;
    for(int i = 0 ; i < N * N ; i++){
        if(C[i] != N)
            break;
        cnt++;
    }
    if(cnt != N * N){
        printf("failed: cnt = %d\n ",cnt);
    }else{
        printf("success\n");
    }
}

// grid = dim3(Bs,Bs), block = dim3(Bs,Bs)
__global__ void block_mul_kernel(
    int Bs,
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
    int lx = threadIdx.y;
    int ly = threadIdx.x;
    int idx = lx * Bs + ly;

    // intialized
    float c = 0;


    for(int bk = 0; bk< n ; bk++){
        A_start_col = bk * Bs;
        B_start_row = bk * Bs;
        
        // load A, B block
        A_block[idx] = d_A[(A_start_row + lx) * N + A_start_col + ly];
        B_block[idx] = d_A[(B_start_row + lx) * N + B_start_col + ly];

        __syncthreads();

        // compute c
        for(int k = 0; k < Bs; k++){
            c += A_block[lx * Bs + k] * B_block[ k * Bs + ly];
        }

        __syncthreads();

    }
    d_C[((bx * Bs) + lx) * N + (by * Bs) + ly ] = c;

}

int main(int argc, char* argv[]){
    assert(argc==2);
    N = atoi(argv[1]);
    create_matrix();
 
    int Bs = 32; // Block size
    int n = N / Bs;
    assert(N % Bs ==0);

    // cudaMalloc
    float *d_A, *d_B, *d_C;
    size_t gmem = N * N *  sizeof(float);
    cudaMalloc((void **)&d_A, gmem);
    cudaMalloc((void **)&d_B, gmem);
    cudaMalloc((void **)&d_C, gmem);
    
    // cuda Memory copy
    cudaMemcpy(d_A, A, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, gmem, cudaMemcpyHostToDevice);

    //launch kernel
    size_t shmem = Bs * Bs * 2 * sizeof(float);
    block_mul_kernel<<<dim3(n,n), dim3(Bs,Bs), shmem>>>(
        Bs, 
        N,
        n,
        d_A,
        d_B,
        d_C
    );

    //cuda Memory copy
    cudaMemcpy(C, d_C, gmem, cudaMemcpyDeviceToHost);

    //free cuda memeory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    correctness_check();

    
    return 0;
}