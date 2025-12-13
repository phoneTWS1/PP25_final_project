#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

int N;
double *A, *B, *C_true;
double *C;
#define Bs 32

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

// grid = dim3(n,n), block = dim3(Bs,Bs)
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
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int idx = lx * Bs + ly;

    // intialized
    double c = 0;

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
    C = (double *)malloc(N * N * sizeof(double));

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

    correctness_check(C_true, C, N);

    
    return 0;
}