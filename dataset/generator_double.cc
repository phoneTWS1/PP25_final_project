#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- 輔助函式：標準矩陣乘法 C = A * B ---
void matmul(const double *a, const double *b, double *c, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            double s = 0.0f;
            for(int k=0; k<n; k++){
                s += a[i*n + k] * b[k*n + j];
            }
            c[i*n + j] = s;
        }
    }
}

static void usage(const char *p){
    fprintf(stderr, "usage: %s N\n", p);
    fprintf(stderr, "  Generates three binary files: A_N, B_N, and C_N (double32, row-major).\n");
    fprintf(stderr, "  C_N contains the result of A_N * B_N.\n");
}

int main(int argc, char **argv){
    if(argc != 2){ usage(argv[0]); return 1; }
    int N = atoi(argv[1]);
    if(N <= 0){ fprintf(stderr, "N must be positive\n"); return 1; }

    size_t sz = (size_t)N * (size_t)N;
    
    // 1. 分配三個獨立的緩衝區
    double *A = (double*)malloc(sz * sizeof(double));
    double *B = (double*)malloc(sz * sizeof(double));
    double *C = (double*)malloc(sz * sizeof(double));
    
    if(!A || !B || !C){ 
        perror("malloc failed for A, B, or C"); 
        free(A); free(B); free(C);
        return 1; 
    }

    // 初始化隨機數種子
    srand((unsigned)time(NULL));
    
    // 2. 獨立地填充 A 和 B，確保它們是隨機且不同的
    for(size_t i=0; i<sz; i++){
        A[i] = (double)rand() / (double)RAND_MAX; // A 是一個隨機矩陣
        B[i] = (double)rand() / (double)RAND_MAX; // B 是另一個不同的隨機矩陣
    }

    // 3. 計算 C = A * B
    printf("Calculating C = A * B (N=%d)... ", N);
    fflush(stdout);
    clock_t start = clock();
    matmul(A, B, C, N);
    clock_t end = clock();
    printf("Done (%.3f s)\n", (double)(end - start) / CLOCKS_PER_SEC);


    // 4. 打開檔案
    char fnameA[256], fnameB[256], fnameC[256];
    snprintf(fnameA, sizeof(fnameA), "A_d_%d", N);
    snprintf(fnameB, sizeof(fnameB), "B_d_%d", N);
    snprintf(fnameC, sizeof(fnameC), "C_d_%d", N);

    FILE *fa = fopen(fnameA, "wb");
    FILE *fb = fopen(fnameB, "wb");
    FILE *fc = fopen(fnameC, "wb");
    
    if(!fa || !fb || !fc){ 
        perror("fopen failed for A, B, or C"); 
        fclose(fa); fclose(fb); fclose(fc); 
        free(A); free(B); free(C); 
        return 1; 
    }

    // 5. 寫入檔案
    if(fwrite(A, sizeof(double), sz, fa) != sz) { perror("fwrite A"); }
    if(fwrite(B, sizeof(double), sz, fb) != sz) { perror("fwrite B"); }
    if(fwrite(C, sizeof(double), sz, fc) != sz) { perror("fwrite C"); }
    
    // 6. 清理
    fclose(fa); fclose(fb); fclose(fc);
    free(A); free(B); free(C);

    printf("Successfully wrote input: %s, %s and answer: %s (binary double32, row-major)\n", fnameA, fnameB, fnameC);
    return 0;
}