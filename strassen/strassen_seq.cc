#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

static int N;
static float *A, *B, *C_result, *C_true; 
static float *A00, *A01, *A10, *A11; 
static float *B00, *B01, *B10, *B11;
static float *m1,*m2, *m3, *m4, *m5, *m6, *m7;
static float *tmp_arr[10]; 

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

void divide(int n2){
    size_t tile_size = n2 * n2 * sizeof(float);
    
    A00 = (float*)calloc(n2 * n2, sizeof(float)); 
    A01 = (float*)calloc(n2 * n2, sizeof(float)); 
    A10 = (float*)calloc(n2 * n2, sizeof(float)); 
    A11 = (float*)calloc(n2 * n2, sizeof(float));
    B00 = (float*)calloc(n2 * n2, sizeof(float)); 
    B01 = (float*)calloc(n2 * n2, sizeof(float)); 
    B10 = (float*)calloc(n2 * n2, sizeof(float)); 
    B11 = (float*)calloc(n2 * n2, sizeof(float));
    m1 = (float*)calloc(n2 * n2, sizeof(float)); 
    m2 = (float*)calloc(n2 * n2, sizeof(float)); 
    m3 = (float*)calloc(n2 * n2, sizeof(float)); 
    m4 = (float*)calloc(n2 * n2, sizeof(float)); 
    m5 = (float*)calloc(n2 * n2, sizeof(float)); 
    m6 = (float*)calloc(n2 * n2, sizeof(float)); 
    m7 = (float*)calloc(n2 * n2, sizeof(float));

    for(int i=0;i<10;i++) tmp_arr[i] = (float*)calloc(n2 * n2, sizeof(float));

    size_t n2row = n2 * sizeof(float);
    for(int i=0;i<n2;i++){
        memcpy(A00 + i*n2, A + i*N, n2row);
        memcpy(A01 + i*n2, A + i*N + n2, n2row);
        memcpy(A10 + i*n2, A + (n2 + i)*N, n2row);
        memcpy(A11 + i*n2, A + (n2 + i)*N + n2, n2row);

        memcpy(B00 + i*n2, B + i*N, n2row);
        memcpy(B01 + i*n2, B + i*N + n2, n2row);
        memcpy(B10 + i*n2, B + (n2 + i)*N, n2row);
        memcpy(B11 + i*n2, B + (n2 + i)*N + n2, n2row);
    }
}

void merge(int n2){
    size_t n2row = n2 * sizeof(float);
    for(int i=0;i<n2;i++){
        memcpy(C_result + i*N, A00 + i*n2, n2row);
        memcpy(C_result + i*N + n2, A01 + i*n2, n2row);
        memcpy(C_result + (n2 + i)*N, A10 + i*n2, n2row);
        memcpy(C_result + (n2 + i)*N + n2, A11 + i*n2, n2row);
    }
}

void add_matrix(const float *a, const float *b, float *c, int n, float alpha, float beta){
    int sz = n*n;
    for(int i=0;i<sz;i++){
        c[i] = alpha * a[i] + beta * b[i];
    }
}

void matmul(const float *a, const float *b, float *c, int n){
    // 初始化 c 為零 (重要!)
    memset(c, 0, n * n * sizeof(float));
    
    // 標準矩陣乘法
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            float sum = 0.0f;
            for(int k=0;k<n;k++){
                sum += a[i*n + k] * b[k*n + j];
            }
            c[i*n + j] = sum;
        }
    }
}

void correctness_check(const float *C_true, const float *C_result, int N){
    int mismatch_count = 0;
    float tol = 5e-3f;
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

void cleanup_memory() {
    if (A) free(A);
    if (B) free(B);
    if (C_true) free(C_true);
    if (C_result) free(C_result);
    
    if (A00) free(A00); if (A01) free(A01); if (A10) free(A10); if (A11) free(A11);
    if (B00) free(B00); if (B01) free(B01); if (B10) free(B10); if (B11) free(B11);
    
    if (m1) free(m1); if (m2) free(m2); if (m3) free(m3); if (m4) free(m4); 
    if (m5) free(m5); if (m6) free(m6); if (m7) free(m7);
    
    for(int i=0; i<10; i++) {
        if (tmp_arr[i]) free(tmp_arr[i]);
    }
}

void print_matrix(const float *matrix, int N_dim, const char *name) {
    if (matrix == NULL) {
        printf("Matrix %s is NULL. Cannot print.\n", name);
        return;
    }
    
    printf("\n--- Matrix %s (%d x %d) ---\n", name, N_dim, N_dim);
    int display_N = (N_dim > 8) ? 8 : N_dim;
    
    for (int i = 0; i < display_N; i++) {
        for (int j = 0; j < display_N; j++) {
            float value = matrix[i * N_dim + j];
            printf("%8.4f ", value); 
        }
        if (N_dim > 8) printf("...");
        printf("\n");
    }
    if (N_dim > 8) printf("...\n");
    printf("-------------------------------\n");
}

int main(int argc, char **argv){
    if(argc != 5){
        fprintf(stderr, "usage: %s N A_filename B_filename C_true_filename\n", argv[0]);
        fprintf(stderr, "N must be even for 1-level Strassen\n");
        return 1;
    }
    
    N = atoi(argv[1]);
    const char *a_filename = argv[2];
    const char *b_filename = argv[3];
    const char *c_true_filename = argv[4];

    if(N <= 0 || (N % 2) != 0){
        fprintf(stderr, "Error: N=%d must be positive and even.\n", N);
        return 1;
    }

    printf("Loading data for N=%d...\n", N);
    load_matrix(&A, N, a_filename);
    load_matrix(&B, N, b_filename);
    load_matrix(&C_true, N, c_true_filename); 
    
    C_result = (float*)calloc(N * N, sizeof(float));
    if (C_result == NULL) {
        perror("Host malloc failed for C_result");
        cleanup_memory();
        return 1;
    }
    
    int n2 = N/2;
    divide(n2);

    // Strassen 的 10 個加減法
    add_matrix(A00, A11, tmp_arr[0], n2, 1.0f, 1.0f);      // S1
    add_matrix(B00, B11, tmp_arr[1], n2, 1.0f, 1.0f);      // S2
    add_matrix(A10, A11, tmp_arr[2], n2, 1.0f, 1.0f);      // S3
    add_matrix(B01, B11, tmp_arr[3], n2, 1.0f, -1.0f);     // S4
    add_matrix(B10, B00, tmp_arr[4], n2, 1.0f, -1.0f);     // S5
    add_matrix(A00, A01, tmp_arr[5], n2, 1.0f, 1.0f);      // S6
    add_matrix(A10, A00, tmp_arr[6], n2, 1.0f, -1.0f);     // S7
    add_matrix(B00, B01, tmp_arr[7], n2, 1.0f, 1.0f);      // S8
    add_matrix(A01, A11, tmp_arr[8], n2, 1.0f, -1.0f);     // S9
    add_matrix(B10, B11, tmp_arr[9], n2, 1.0f, 1.0f);      // S10

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // 7 次乘法
    matmul(tmp_arr[0], tmp_arr[1], m1, n2); // M1 = (A00+A11)*(B00+B11)
    matmul(tmp_arr[2], B00, m2, n2);        // M2 = (A10+A11)*B00
    matmul(A00, tmp_arr[3], m3, n2);        // M3 = A00*(B01-B11)
    matmul(A11, tmp_arr[4], m4, n2);        // M4 = A11*(B10-B00)
    matmul(tmp_arr[5], B11, m5, n2);        // M5 = (A00+A01)*B11
    matmul(tmp_arr[6], tmp_arr[7], m6, n2); // M6 = (A10-A00)*(B00+B01)
    matmul(tmp_arr[8], tmp_arr[9], m7, n2); // M7 = (A01-A11)*(B10+B11)

    // 組合結果
    // C00 = M1 + M4 - M5 + M7
    add_matrix(m1, m7, tmp_arr[1], n2, 1.0f, 1.0f);
    add_matrix(m4, m5, tmp_arr[2], n2, 1.0f, -1.0f);
    add_matrix(tmp_arr[1], tmp_arr[2], A00, n2, 1.0f, 1.0f);
    
    // C01 = M3 + M5
    add_matrix(m3, m5, A01, n2, 1.0f, 1.0f);

    // C10 = M2 + M4
    add_matrix(m2, m4, A10, n2, 1.0f, 1.0f);

    // C11 = M1 - M2 + M3 + M6
    add_matrix(m1, m2, tmp_arr[4], n2, 1.0f, -1.0f);
    add_matrix(m3, m6, tmp_arr[5], n2, 1.0f, 1.0f);
    add_matrix(tmp_arr[4], tmp_arr[5], A11, n2, 1.0f, 1.0f);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)*1e-9;
    printf("Compute time : %.6f s\n", elapsed);

    merge(n2);
    correctness_check(C_true, C_result, N);
    cleanup_memory();
    
    return 0;
}