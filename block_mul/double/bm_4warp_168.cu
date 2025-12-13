#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

int N;
double *A, *B, *C_true;
double *C;
#define Bs 32

void load_matrix(double**, int, const char *filename);
void correctness_check(const double*, const double*, int);
void show(double*, int);


__inline__ __device__ double warpReduceSum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
// every wrap is reposible for 4 row of a C tile
// every thread is reposible for 4 * 2 elememts
// grid = dim3(n,n), block = dim3(16,8)
__global__ void block_mul_kernel(
    int N,
    int n,
    double *d_A,
    double *d_B,
    double *d_C
){
    extern __shared__ double share[];
    double *A_block = share;
    double *B_block = share + Bs * Bs;

    // block index 
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int A_start_row = bx * Bs;
    int A_start_col;
    int B_start_row;
    int B_start_col = by * Bs;
    
    // block local index
    int lx = threadIdx.y; // threadIdx.y = 0...8, 
    int ly = threadIdx.x; // threadIdx.x = 0...15
    int stride = blockDim.y;

    int grow1 = lx + stride;
    int grow2 = grow1 + stride;
    int grow3 = grow2 + stride;

    int row0 = lx * Bs;
    int row1 = grow1 * Bs;
    int row2 = grow2 * Bs;
    int row3 = grow3 * Bs;


    int col0 = ly;
    int col1 = ly + blockDim.x;


    double c00=0, c01=0, c10=0, c11=0, c20=0, c21=0, c30=0, c31=0;
    double a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;

   // compute c
    for(int bk = 0; bk< n ; bk++){
        A_start_col = bk * Bs;
        B_start_row = bk * Bs;
        
        // load A, B block
        A_block[row0 + col0] = d_A[(A_start_row + lx) * N + A_start_col + col0];
        A_block[row0 + col1] = d_A[(A_start_row + lx) * N + A_start_col + col1];
        A_block[row1 + col0] = d_A[(A_start_row + grow1) * N + A_start_col + col0];
        A_block[row1 + col1] = d_A[(A_start_row + grow1) * N + A_start_col + col1];
        A_block[row2 + col0] = d_A[(A_start_row + grow2) * N + A_start_col + col0];
        A_block[row2 + col1] = d_A[(A_start_row + grow2) * N + A_start_col + col1];
        A_block[row3 + col0] = d_A[(A_start_row + grow3) * N + A_start_col + col0];
        A_block[row3 + col1] = d_A[(A_start_row + grow3) * N + A_start_col + col1];
        
        B_block[row0 + col0] = d_B[(B_start_row + lx) * N + B_start_col + col0];
        B_block[row0 + col1] = d_B[(B_start_row + lx) * N + B_start_col + col1];
        B_block[row1 + col0] = d_B[(B_start_row + grow1) * N + B_start_col + col0];
        B_block[row1 + col1] = d_B[(B_start_row + grow1) * N + B_start_col + col1];
        B_block[row2 + col0] = d_B[(B_start_row + grow2) * N + B_start_col + col0];
        B_block[row2 + col1] = d_B[(B_start_row + grow2) * N + B_start_col + col1];
        B_block[row3 + col0] = d_B[(B_start_row + grow3) * N + B_start_col + col0];
        B_block[row3 + col1] = d_B[(B_start_row + grow3) * N + B_start_col + col1];

        __syncthreads();
        
        #pragma unroll 32
        for(int k = 0 ; k < Bs ; k ++){
            b0 = B_block[k * Bs + col0];
            b1 = B_block[k * Bs + col1];
            
            a0 = A_block[row0 + k];
            a1 = A_block[row1 + k];
            a2 = A_block[row2 + k];
            a3 = A_block[row3 + k];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
            c20 += a2 * b0;
            c21 += a2 * b1;
            c30 += a3 * b0;
            c31 += a3 * b1;
        
        }
        __syncthreads();
    }

    //write back
    d_C[(A_start_row + lx) * N + B_start_col + col0] = c00;
    d_C[(A_start_row + lx) * N + B_start_col + col1] = c01;
    d_C[(A_start_row + grow1) * N + B_start_col + col0] = c10;
    d_C[(A_start_row + grow1) * N + B_start_col + col1] = c11;
    d_C[(A_start_row + grow2) * N + B_start_col + col0] = c20;
    d_C[(A_start_row + grow2) * N + B_start_col + col1] = c21;
    d_C[(A_start_row + grow3) * N + B_start_col + col0] = c30;
    d_C[(A_start_row + grow3) * N + B_start_col + col1] = c31;


}

int main(int argc, char* argv[]){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    assert(argc==5);
    N = atoi(argv[1]);
    const char *a_filename = argv[2];
    const char *b_filename = argv[3];
    const char *c_true_filename = argv[4];
    load_matrix(&A, N, a_filename);
    load_matrix(&B, N, b_filename);
    load_matrix(&C_true, N, c_true_filename); 
    C = (double*)malloc(N * N * sizeof(double));

    //int Bs = 32; // Block size
    int n = N / Bs;
    assert(N % Bs ==0);

    // cudaMalloc
    double *d_A, *d_B, *d_C;
    size_t gmem = N * N *  sizeof(double);
    assert(gmem * 3 < prop.totalGlobalMem);
    cudaMalloc((void **)&d_A, gmem);
    cudaMalloc((void **)&d_B, gmem);
    cudaMalloc((void **)&d_C, gmem);
    
    // cuda Memory copy
    cudaMemcpy(d_A, A, gmem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, gmem, cudaMemcpyHostToDevice);


    //launch kernel
    size_t shmem = Bs * Bs * 2 * sizeof(double);
    block_mul_kernel<<<dim3(n,n), dim3(16,8), shmem>>>(
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

    correctness_check(C_true, C, N);

    
    return 0;
}

void load_matrix(double **mat_ptr, int N_dim, const char *filename) {
    long long size = (long long)N_dim * N_dim;
    size_t bytes_f = size * sizeof(float);
    size_t bytes_d = size * sizeof(double);

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // read as float then convert to double (dataset produced float32)
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
    double tol = 5e-3f;
    long long sz = (long long)N * N;
    double max_error = 0.0f;
    
    for(long long i=0; i < sz; i++){
        double error = fabsf(C_result[i] - C_true[i]);
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

void show(double* M, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N ;j++){
            printf("%.0f, ", M[i * N+ j]);
        }
        printf("\n");
    }
    printf("\n");
}