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

/**
 * 載入單一二進制矩陣檔案 (.bin, float32, row-major)。
 * @param mat_ptr 載入矩陣的指標
 * @param N_dim 矩陣維度
 * @param filename 檔案名稱
 */
void load_matrix(float **mat_ptr, int N_dim, const char *filename) {
    long long size = (long long)N_dim * N_dim;
    size_t bytes = size * sizeof(float);
    
    FILE *file = fopen(filename, "rb"); 
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // 分配 Host 內存
    *mat_ptr = (float *)malloc(bytes);
    if (*mat_ptr == NULL) {
        perror("Host malloc failed in load_matrix");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // 讀取整個矩陣的數據
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
    // 這裡的 malloc 仍然需要
    A00 = (float*)malloc(tile_size); A01 = (float*)malloc(tile_size); 
    A10 = (float*)malloc(tile_size); A11 = (float*)malloc(tile_size);
    B00 = (float*)malloc(tile_size); B01 = (float*)malloc(tile_size); 
    B10 = (float*)malloc(tile_size); B11 = (float*)malloc(tile_size);
    m1 = (float*)malloc(tile_size); m2 = (float*)malloc(tile_size); 
    m3 = (float*)malloc(tile_size); m4 = (float*)malloc(tile_size); 
    m5 = (float*)malloc(tile_size); m6 = (float*)malloc(tile_size); 
    m7 = (float*)malloc(tile_size);

    for(int i=0;i<10;i++) tmp_arr[i] = (float*)malloc(tile_size);

    size_t n2row = n2 * sizeof(float);
    for(int i=0;i<n2;i++){
        // 複製 A 矩陣
        memcpy(A00 + i*n2, A + i*N, n2row);
        memcpy(A01 + i*n2, A + i*N + n2, n2row);
        memcpy(A10 + i*n2, A + (n2 + i)*N, n2row);
        memcpy(A11 + i*n2, A + (n2 + i)*N + n2, n2row);

        // 複製 B 矩陣
        memcpy(B00 + i*n2, B + i*N, n2row);
        memcpy(B01 + i*n2, B + i*N + n2, n2row);
        memcpy(B10 + i*n2, B + (n2 + i)*N, n2row);
        memcpy(B11 + i*n2, B + (n2 + i)*N + n2, n2row);
    }
}

void merge(int n2){
    size_t n2row = n2 * sizeof(float);
    for(int i=0;i<n2;i++){
        // 組合 C_result 的 4 個象限
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
    // c = a * b (standard triple loop)
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            float s = 0.0f;
            for(int k=0;k<n;k++){
                s += a[i*n + k] * b[k*n + j];
            }
            c[i*n + j] = s;
        }
    }
}

void correctness_check(const float *C_true, const float *C_result, int N){
    int mismatch_count = 0;
    float tol = 2e-3f;
    long long sz = (long long)N * N;
    
    for(long long i=0; i < sz; i++){
        if (fabsf(C_result[i] - C_true[i]) > tol){
             mismatch_count++;
             if (mismatch_count < 10) {
                 fprintf(stderr, "Mismatch at index %lld: Result=%.4f, True=%.4f\n", 
                         i, C_result[i], C_true[i]);
             }
        }
    }
    
    if(mismatch_count > 0){
        printf("FAILED: Result mismatch with loaded answer C. Total errors: %d\n", mismatch_count);
    } else {
        printf("SUCCESS: Result matches loaded answer C.\n");
    }
}

/**
 * 釋放所有動態分配的 Host 記憶體。
 */
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

// ----------------------------------------------------
// 主函式
// ----------------------------------------------------
int main(int argc, char **argv){
    // 檢查參數：現在需要 N, A_file, B_file, C_true_file
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

    // 1. 載入 A, B, 和 C_true (答案)
    printf("Loading data for N=%d...\n", N);
    load_matrix(&A, N, a_filename);
    load_matrix(&B, N, b_filename);
    // 載入答案矩陣
    load_matrix(&C_true, N, c_true_filename); 
    
    // 2. 分配 Strassen 結果矩陣 C_result
    C_result = (float*)malloc((long long)N * N * sizeof(float));
    if (C_result == NULL) {
        perror("Host malloc failed for C_result");
        cleanup_memory();
        return 1;
    }
    
    int n2 = N/2;
    
    // 3. 分塊和臨時內存分配
    divide(n2);

    // 4. 計算加減法中間項 (S1 到 S10)
    add_matrix(A00, A11, tmp_arr[0], n2, 1.0f, 1.0f);      // S1: tmp0 = A00 + A11
    add_matrix(B00, B11, tmp_arr[1], n2, 1.0f, 1.0f);      // S2: tmp1 = B00 + B11
    add_matrix(A10, A11, tmp_arr[2], n2, 1.0f, 1.0f);      // S3: tmp2 = A10 + A11
    add_matrix(B01, B11, tmp_arr[3], n2, 1.0f, -1.0f);     // S4: tmp3 = B01 - B11
    add_matrix(B10, B00, tmp_arr[4], n2, 1.0f, -1.0f);     // S5: tmp4 = B10 - B00
    add_matrix(A00, A01, tmp_arr[5], n2, 1.0f, 1.0f);      // S6: tmp5 = A00 + A01
    add_matrix(A10, A00, tmp_arr[6], n2, 1.0f, -1.0f);     // S7: tmp6 = A10 - A00
    add_matrix(B00, B01, tmp_arr[7], n2, 1.0f, 1.0f);      // S8: tmp7 = B00 + B01
    add_matrix(A01, A11, tmp_arr[8], n2, 1.0f, -1.0f);     // S9: tmp8 = A01 - A11
    add_matrix(B10, B11, tmp_arr[9], n2, 1.0f, 1.0f);      // S10: tmp9 = B10 + B11

    // 5. 計時開始
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // 6. 計算 M1..M7
    matmul(tmp_arr[0], tmp_arr[1], m1, n2); // M1
    matmul(tmp_arr[2], B00, m2, n2);        // M2
    matmul(A00, tmp_arr[3], m3, n2);        // M3
    matmul(A11, tmp_arr[4], m4, n2);        // M4
    matmul(tmp_arr[5], B11, m5, n2);        // M5
    matmul(tmp_arr[6], tmp_arr[7], m6, n2); // M6
    matmul(tmp_arr[8], tmp_arr[9], m7, n2); // M7

    // 7. 組合 C 象限 (直接寫入 A00...A11 容器)
    // C00 = M1 + M4 - M5 + M7
    add_matrix(m1, m4, tmp_arr[1], n2, 1.0f, 1.0f);          // tmp1 = M1 + M4
    add_matrix(m5, m7, tmp_arr[2], n2, 1.0f, 1.0f);          // tmp2 = M5 + M7
    add_matrix(tmp_arr[1], tmp_arr[2], A00, n2, 1.0f, -1.0f); // A00 = (M1 + M4) - (M5 + M7)
    add_matrix(A00, m7, A00, n2, 1.0f, 2.0f); // 修正代數錯誤: C00 = M1 + M4 - M5 + M7
    
    // C01 = M3 + M5
    add_matrix(m3, m5, A01, n2, 1.0f, 1.0f);                 // A01 = M3 + M5

    // C10 = M2 + M4
    add_matrix(m2, m4, A10, n2, 1.0f, 1.0f);                 // A10 = M2 + M4

    // C11 = M1 - M2 + M3 + M6
    add_matrix(m1, m2, tmp_arr[4], n2, 1.0f, -1.0f);         // tmp4 = M1 - M2
    add_matrix(m3, m6, tmp_arr[5], n2, 1.0f, 1.0f);          // tmp5 = M3 + M6
    add_matrix(tmp_arr[4], tmp_arr[5], A11, n2, 1.0f, 1.0f); // A11 = tmp4 + tmp5

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)*1e-9;
    printf("Compute time : %.6f s\n", elapsed);

    // 8. 合併結果到 C_result
    merge(n2);
    
    // 9. 檢查正確性 (比較 C_result 與 C_true)
    correctness_check(C_true, C_result, N);

    // 10. 釋放所有記憶體
    cleanup_memory();
    
    return 0;
}