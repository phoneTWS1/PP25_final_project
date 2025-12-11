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
// every wrap is reposible for 4 row of a C tile
// every thread is reposible for 4 * 4 (same column)
// grid = dim3(n,n), block = dim3(8,8)
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
    int lx = threadIdx.y; // threadIdx.y = 0...8, 
    int ly = threadIdx.x; // threadIdx.x = 0...8
    int stride = blockDim.y;


    float c[4][4];
    #pragma unroll 4
    for(int i=0; i<4; i++){
        #pragma unroll 4
        for(int j=0 ; j< 4; j++){
            c[i][j] = 0;
        }
    }

    float b[4], a[4];
    
   // compute c
    for(int bk = 0; bk< n ; bk++){
        A_start_col = bk * Bs;
        B_start_row = bk * Bs;
        
        // load A, B block
        #pragma unroll 4
        for(int i = 0 ; i < 4; i++){
            #pragma unroll 4
            for(int j = 0; j < 4; j++){
                A_block[(lx + i * stride) * Bs + (ly + j * stride)] = d_A[(A_start_row + lx + i * stride) * N + (A_start_col + ly + j * stride)];
                B_block[(lx + i * stride) * Bs + (ly + j * stride)] = d_B[(B_start_row + lx + i * stride) * N + (B_start_col + ly + j * stride)];
            }
        }

        __syncthreads();

        #pragma unroll 32
        for(int k = 0 ; k < Bs ; k ++){
            // load a's
            #pragma unroll 4
            for(int i = 0 ; i < 4 ; i++){
                a[i] = A_block[(lx + i * stride) * Bs + k];
            }

            // load b's
            #pragma unroll 4
            for(int j = 0 ; j < 4 ; j++){
                b[j] = B_block[k * Bs + ly + j * stride];
            }
            
            #pragma unroll 4
            for(int i = 0; i < 4 ; i++){
                #pragma unroll 4
                for(int j = 0 ; j < 4 ; j ++){
                    c[i][j] += a[i] * b[j];
                }
            }

        }
        __syncthreads();

    }

    //write back
    #pragma unroll 4
    for(int i = 0; i < 4 ; i++){
        #pragma unroll 4
        for(int j = 0 ; j < 4 ; j++){
            d_C[(A_start_row + lx + i * stride) * N + (B_start_col + ly + j * stride)] = c[i][j];
        }
    }
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
    block_mul_kernel<<<dim3(n,n), dim3(8,8), shmem>>>(
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