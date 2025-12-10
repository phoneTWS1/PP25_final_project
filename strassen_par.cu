#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <string.h>


int N;
float *A, *B;
float *C;

float *A00, *A01, *A10, *A11;
float *B00, *B01, *B10, *B11;
float *m1,*m2, *m3, *m4, *m5, *m6, *m7;

void create_matrix(void);
void correctness_check(void);
void divide(int);
void merge(int);

void show_block(float*, int);


__global__ void block_mul_kernel(
    float *d_A,
    float *d_B,
    float *d_C,
    int N,
    int n,
    int Bs
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
        B_block[idx] = d_B[(B_start_row + lx) * N + B_start_col + ly];

        __syncthreads();

        // compute c
        #pragma unroll 32
        for(int k = 0; k < Bs; k++){
            c += A_block[lx * Bs + k] * B_block[ k * Bs + ly];
        }

        __syncthreads();

    }
    d_C[((bx * Bs) + lx) * N + (by * Bs) + ly ] = c;

}

// grid = (n,n) block = (Bs,Bs)
__global__ void block_add_matrix(
    float *a,
    float *b,
    float *c,
    int n2,
    int Bs,
    float alpha,
    float beta
){
    //global index
    int x = blockIdx.y * Bs + threadIdx.y;
    int y = blockIdx.x * Bs + threadIdx.x;
    int idx = x * n2 + y;
    c[idx] = alpha * a[idx] + beta * b[idx];
}

int main(int argc, char* argv[]){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    float ms = 0;
   //float cmp = 0, cpy = 0, div = 0, merge = 0 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    assert(argc==2);
    N = atoi(argv[1]);
    

    cudaEventRecord(start);
    create_matrix();


    // divide tile
    int n2 = N/2;
    divide(n2);

    //cudaMalloc
    float *d_A00, *d_A01, *d_A10, *d_A11; 
    float *d_B00, *d_B01, *d_B10, *d_B11;
    float *d_m1, *d_m2, *d_m3, *d_m4, *d_m5, *d_m6, *d_m7;
    float *tmp[10];
    size_t gmem = n2 * n2 * sizeof(float);
    assert(gmem * 25 < prop.totalGlobalMem);
    cudaMalloc((void **) &d_A00, gmem);
    cudaMalloc((void **) &d_A01, gmem);
    cudaMalloc((void **) &d_A10, gmem);
    cudaMalloc((void **) &d_A11, gmem);

    cudaMalloc((void **) &d_B00, gmem);
    cudaMalloc((void **) &d_B01, gmem);
    cudaMalloc((void **) &d_B10, gmem);
    cudaMalloc((void **) &d_B11, gmem);

    cudaMemcpy(d_A00, A00, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A01, A01, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A10, A10, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A11, A11, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B00, B00, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B01, B01, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B10, B10, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B11, B11, gmem, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_m1, gmem);
    cudaMalloc((void **) &d_m2, gmem);
    cudaMalloc((void **) &d_m3, gmem);
    cudaMalloc((void **) &d_m4, gmem);
    cudaMalloc((void **) &d_m5, gmem);
    cudaMalloc((void **) &d_m6, gmem);
    cudaMalloc((void **) &d_m7, gmem);

    for(int i = 0; i<10; i++){
        cudaMalloc((void**) &tmp[i], gmem);
    }
    
    //compute ms
    int Bs = 32;
    int n = n2/Bs;

    cudaEventRecord(start);
    // m1
    block_add_matrix<<<dim3(n,n),dim3(Bs,Bs)>>>(d_A00, d_A11, tmp[0], n2, Bs, 1.0f, 1.0f);
    block_add_matrix<<<dim3(n,n),dim3(Bs,Bs)>>>(d_B00, d_B11, tmp[1], n2, Bs, 1.0f, 1.0f);
    //m2
    block_add_matrix<<<dim3(n,n),dim3(Bs,Bs)>>>(d_A10, d_A11, tmp[2], n2, Bs, 1.0f, 1.0f);
    //m3
    block_add_matrix<<<dim3(n,n),dim3(Bs,Bs)>>>(d_B01, d_B11, tmp[3], n2, Bs, 1.0f, -1.0f);
    //m4
    block_add_matrix<<<dim3(n,n),dim3(Bs,Bs)>>>(d_B10, d_B00, tmp[4], n2, Bs, 1.0f, -1.0f);
    //m5
    block_add_matrix<<<dim3(n,n),dim3(Bs,Bs)>>>(d_A00, d_A01, tmp[5], n2, Bs, 1.0f, 1.0f);
    //m6
    block_add_matrix<<<dim3(n,n),dim3(Bs,Bs)>>>(d_A10, d_A00, tmp[6], n2, Bs, 1.0f, -1.0f);
    block_add_matrix<<<dim3(n,n),dim3(Bs,Bs)>>>(d_B00, d_B01, tmp[7], n2, Bs, 1.0f, 1.0f);
    //m7
    block_add_matrix<<<dim3(n,n),dim3(Bs,Bs)>>>(d_A01, d_A11, tmp[8], n2, Bs, 1.0f, -1.0f);
    block_add_matrix<<<dim3(n,n),dim3(Bs,Bs)>>>(d_B10, d_B11, tmp[9], n2, Bs, 1.0f, 1.0f);

    

    //m1
    size_t shmem = Bs * Bs * 2 * sizeof(float);
    block_mul_kernel<<<dim3(n,n), dim3(Bs,Bs), shmem>>>(tmp[0], tmp[1], d_m1, n2, n, Bs);
    //m2
    block_mul_kernel<<<dim3(n,n), dim3(Bs,Bs), shmem>>>(tmp[2], d_B00, d_m2, n2, n, Bs);
    //m3
    block_mul_kernel<<<dim3(n,n), dim3(Bs,Bs), shmem>>>(d_A00, tmp[3], d_m3, n2, n, Bs);
    //m4
    block_mul_kernel<<<dim3(n,n), dim3(Bs,Bs), shmem>>>(d_A11, tmp[4], d_m4, n2, n, Bs);
    //m5
    block_mul_kernel<<<dim3(n,n), dim3(Bs,Bs), shmem>>>(tmp[5], d_B11, d_m5, n2, n, Bs);
    //m6
    block_mul_kernel<<<dim3(n,n), dim3(Bs,Bs), shmem>>>(tmp[6], tmp[7], d_m6, n2, n, Bs);
    //m7
    block_mul_kernel<<<dim3(n,n), dim3(Bs,Bs), shmem>>>(tmp[8], tmp[9], d_m7, n2, n, Bs);
    

    //c00
    block_add_matrix<<<dim3(n,n), dim3(Bs,Bs)>>>(d_m1, d_m4, tmp[1], n2, Bs, 1.0f, 1.0f);
    block_add_matrix<<<dim3(n,n), dim3(Bs,Bs)>>>(d_m5, d_m7, tmp[2], n2, Bs, 1.0f, 1.0f);
    block_add_matrix<<<dim3(n,n), dim3(Bs,Bs)>>>(tmp[1], tmp[2], tmp[0], n2, Bs, 1.0f, -1.0f);
    //C01
    block_add_matrix<<<dim3(n,n), dim3(Bs,Bs)>>>(d_m3, d_m5, tmp[1], n2, Bs, 1.0f, 1.0f);
    //C10
    block_add_matrix<<<dim3(n,n), dim3(Bs,Bs)>>>(d_m2, d_m4, tmp[2], n2, Bs, 1.0f, 1.0f);
    //C11
    block_add_matrix<<<dim3(n,n), dim3(Bs,Bs)>>>(d_m1, d_m2, tmp[4], n2, Bs, 1.0f, -1.0f);
    block_add_matrix<<<dim3(n,n), dim3(Bs,Bs)>>>(d_m3, d_m6, tmp[5], n2, Bs, 1.0f, 1.0f);
    block_add_matrix<<<dim3(n,n), dim3(Bs,Bs)>>>(tmp[4], tmp[5], tmp[3], n2, Bs, 1.0f, 1.0f);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("compute take : %.3f s\n", ms * 1e-3);


    // cuda compy back
    cudaMemcpy(A00, tmp[0], gmem, cudaMemcpyDeviceToHost);
    cudaMemcpy(A01, tmp[1], gmem, cudaMemcpyDeviceToHost);
    cudaMemcpy(A10, tmp[2], gmem, cudaMemcpyDeviceToHost);
    cudaMemcpy(A11, tmp[3], gmem, cudaMemcpyDeviceToHost);
    merge(n2);
    
    correctness_check();
    
    
    return 0;
}

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


void divide(int n2){
    size_t tile_size = n2 * n2 * sizeof(float);
    A00 = (float*)malloc(tile_size);
    A01 = (float*)malloc(tile_size);
    A10 = (float*)malloc(tile_size);
    A11 = (float*)malloc(tile_size);

    B00 = (float*)malloc(tile_size);
    B01 = (float*)malloc(tile_size);
    B10 = (float*)malloc(tile_size);
    B11 = (float*)malloc(tile_size);

    m1 = (float*)malloc(tile_size);
    m2 = (float*)malloc(tile_size);
    m3 = (float*)malloc(tile_size);
    m4 = (float*)malloc(tile_size);
    m5 = (float*)malloc(tile_size);
    m6 = (float*)malloc(tile_size);
    m7 = (float*)malloc(tile_size);
    
    size_t n2row = n2 * sizeof(float);

    for(int i=0; i < n2 ; i++){
        memcpy(A00 + i * n2, A + i * N, n2row);
        memcpy(A01 + i * n2, A + i * N + n2, n2row);
        memcpy(A10 + i * n2, A + (n2 + i)*N, n2row);
        memcpy(A11 + i * n2, A + (n2 + i)*N + n2, n2row);

        memcpy(B00 + i * n2, B + i * N, n2row);
        memcpy(B01 + i * n2, B + i * N + n2, n2row);
        memcpy(B10 + i * n2, B + (n2 + i)*N, n2row);
        memcpy(B11 + i * n2, B + (n2 + i)*N + n2, n2row);
    }

}
void merge(int n2){
    size_t n2row = n2 * sizeof(float);
    for(int i=0; i<n2; i++){
        memcpy(C + i * N, A00 + i * n2, n2row);
        memcpy(C + i * N + n2, A01 + i * n2 , n2row);
        memcpy(C + (n2 + i)*N, A10 + i * n2, n2row);
        memcpy(C + (n2 + i)*N + n2, A11 + i * n2, n2row);
    }

}
void show_block(float * m, int n2){
    for(int i=0; i < n2; i++){
        for(int j=0; j<n2; j++){
            printf("%.1f, ", m[i *n2 + j]);
        }
        printf("\n");
    }
    printf("\n");
}