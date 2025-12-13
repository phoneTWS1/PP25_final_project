#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <chrono>

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
// every thread is reposible for 4 rows elements (same column)
// grid = dim3(n,n), block = dim3(32,8)
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
    int lx = threadIdx.y; // threadIdx.y = 0...7, 
    int ly = threadIdx.x; // threadIdx.x = 0...31
    int wrap_row = lx * 4;


    double c0=0, c1=0, c2=0, c3=0;

    // compute c
    for(int bk = 0; bk< n ; bk++){
        A_start_col = bk * Bs;
        B_start_row = bk * Bs;
        
        // load A, B block
        A_block[wrap_row * Bs + ly] = d_A[(A_start_row + wrap_row) * N + A_start_col + ly];
        A_block[(wrap_row + 1) * Bs + ly] = d_A[(A_start_row + wrap_row + 1) * N + A_start_col + ly];
        A_block[(wrap_row + 2) * Bs + ly] = d_A[(A_start_row + wrap_row + 2) * N + A_start_col + ly];
        A_block[(wrap_row + 3) * Bs + ly] = d_A[(A_start_row + wrap_row + 3) * N + A_start_col + ly];

        B_block[wrap_row * Bs + ly] = d_B[(B_start_row + wrap_row) * N + B_start_col + ly];
        B_block[(wrap_row + 1) * Bs + ly] = d_B[(B_start_row + wrap_row + 1) * N + B_start_col + ly];
        B_block[(wrap_row + 2) * Bs + ly] = d_B[(B_start_row + wrap_row + 2) * N + B_start_col + ly];
        B_block[(wrap_row + 3) * Bs + ly] = d_B[(B_start_row + wrap_row + 3) * N + B_start_col + ly];

        __syncthreads();
        
        #pragma unroll 32
        for(int k = 0 ; k < Bs ; k ++){
            double b = B_block[k * Bs + ly];
            double a0 = A_block[(wrap_row * Bs) + k];
            double a1 = A_block[(wrap_row + 1) * Bs + k];
            double a2 = A_block[(wrap_row + 2) * Bs + k];
            double a3 = A_block[(wrap_row + 3) * Bs + k];

            // 4FMAs, 5 shared load  =>  0.8 FMAs/load;
            c0 += a0 * b;
            c1 += a1 * b;
            c2 += a2 * b;
            c3 += a3 * b;
        }

        __syncthreads();
    }

    //write back
    d_C[(A_start_row + wrap_row) * N + B_start_col + ly] = c0;
    d_C[(A_start_row + wrap_row + 1) * N + B_start_col + ly] = c1;
    d_C[(A_start_row + wrap_row + 2) * N + B_start_col + ly] = c2;
    d_C[(A_start_row + wrap_row + 3) * N + B_start_col + ly] = c3;
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

    auto chrono_start = std::chrono::steady_clock::now();
    //launch kernel
    size_t shmem = Bs * Bs * 2 * sizeof(double);

    block_mul_kernel<<<dim3(n,n), dim3(Bs,8), shmem>>>(
        N,
        n,
        d_A,
        d_B,
        d_C
    );
    
    cudaStreamSynchronize(0);
    auto chrono_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = chrono_end - chrono_start;
    printf("Total computing time: %.6f s\n", elapsed.count());


    auto chrono_start_copy = std::chrono::steady_clock::now();
    //cuda Memory copy
    cudaMemcpy(C, d_C, gmem, cudaMemcpyDeviceToHost); 
    //show(C,N);
    auto chrono_end_copy = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_copy = chrono_end_copy - chrono_start_copy;
    printf("Data copy back time: %.6f s\n", elapsed_copy.count());

    //free cuda memeory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // correctness_check(C_true, C, N);

    
    return 0;
}

void load_matrix(double **mat_ptr, int N_dim, const char *filename) {
    long long size = (long long)N_dim * N_dim;
    // dataset 是 float32（generator 產生），在此先讀 float 再轉 double
    size_t bytes_f = (size_t)size * sizeof(float);
    size_t bytes_d = (size_t)size * sizeof(double);

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    float *buff = (float*)malloc(bytes_f);
    if (!buff) {
        perror("malloc failed for float buffer");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    size_t read_count = fread(buff, sizeof(float), (size_t)size, file);
    if (read_count != (size_t)size) {
        fprintf(stderr, "Error: Read incomplete from %s. Expected %lld elements, read %zu.\n",
                filename, size, read_count);
        free(buff);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);

    *mat_ptr = (double*)malloc(bytes_d);
    if (!(*mat_ptr)) {
        perror("Host malloc failed in load_matrix");
        free(buff);
        exit(EXIT_FAILURE);
    }
    for (long long i = 0; i < size; ++i) (*mat_ptr)[i] = (double)buff[i];
    free(buff);
}

void correctness_check(const double *C_true, const double *C_result, int N){
    int mismatch_count = 0;
    // 使用相對/絕對誤差容忍度，視需求可調整
    double tol = 5e-3;
    long long sz = (long long)N * N;
    double max_error = 0.0;

    for(long long i=0; i < sz; i++){
        double error = fabs(C_result[i] - C_true[i]);
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