#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

int N;
float *A, *B, *B_T;
float *C;
#define Bs 32

void create_matrix();
void correctness_check();
void transpose();
void show(float* , int);

// grid = dim3(n,n), block = dim3(Bs,Bs)
__global__ void block_mul_kernel(
    int N,
    int n,
    float *d_A,
    float *d_B,
    float *d_C
){
    // block index 
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int start_row = bx * Bs;
    int start_col = by * Bs;
    int inter;
    
    // block local index
    int lx = threadIdx.y;
    int ly = threadIdx.x;
    
    // global index
    int idx = (start_row + lx) * N + start_col + ly;

    for(int bk = 0 ; bk < n ; bk++){
        inter= bk * Bs;

        for(int k = 0; k< Bs ; k++){
            d_C[idx] += d_A[(start_row + lx) * N + (inter + k)] * d_B[(start_col + ly) * N + inter + k];
        }
    }
}

int main(int argc, char* argv[]){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    assert(argc==2);
    N = atoi(argv[1]);

    create_matrix();

    transpose();

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
    cudaMemcpy(d_B, B_T, gmem, cudaMemcpyHostToDevice);

    //launch kernel
    size_t shmem = Bs * Bs * 2 * sizeof(float);
    block_mul_kernel<<<dim3(n,n), dim3(Bs,Bs), shmem>>>(
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

void transpose(){
    B_T = (float *)malloc(N * N * sizeof(float));
    for(int i = 0; i < N ; i++){
        for(int j = 0; j < N ; j ++){
            B_T[j * N + i] = B[i * N + j];
        }
    }
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

void show(float * m, int n2){
    for(int i=0; i < n2; i++){
        for(int j=0; j<n2; j++){
            printf("%.1f, ", m[i *n2 + j]);
        }
        printf("\n");
    }
    printf("\n");
}