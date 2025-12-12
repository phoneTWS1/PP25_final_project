#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static void usage(const char *p){
    fprintf(stderr, "usage: %s N\n", p);
    fprintf(stderr, "  Generates three binary files: A_N, B_N, and C_N (float32, row-major).\n");
    fprintf(stderr, "  Uses CUDA + cuBLAS to compute C = A * B on device.\n");
}

#define CHECK_CUDA(call) do {                                        \
    cudaError_t e = (call);                                          \
    if (e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                          \
    }                                                                 \
} while(0)

#define CHECK_CUBLAS(call) do {                                      \
    cublasStatus_t s = (call);                                       \
    if (s != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); \
        exit(EXIT_FAILURE);                                          \
    }                                                                 \
} while(0)

int main(int argc, char **argv){
    if(argc != 2){ usage(argv[0]); return 1; }
    int N = atoi(argv[1]);
    if(N <= 0){ fprintf(stderr, "N must be positive\n"); return 1; }

    size_t sz = (size_t)N * (size_t)N;
    float *h_A = (float*)malloc(sz * sizeof(float));
    float *h_B = (float*)malloc(sz * sizeof(float));
    float *h_C = (float*)malloc(sz * sizeof(float));
    if(!h_A || !h_B || !h_C){
        perror("malloc failed");
        free(h_A); free(h_B); free(h_C);
        return 1;
    }

    srand((unsigned)time(NULL));
    for(size_t i=0;i<sz;i++){
        h_A[i] = (float)rand() / (float)RAND_MAX;
        h_B[i] = (float)rand() / (float)RAND_MAX;
    }

    // device buffers
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sz * sizeof(float)));

    // copy to device (row-major layout)
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sz * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // compute C = A * B where A,B,C are stored row-major on host/device
    // Trick: call cuBLAS with transposed operands to handle row-major:
    //   C_row = A_row * B_row  <=>  C_col = (C_row)^T = B_col * A_col
    // So perform: C = (B^T) * (A^T) with opA=transpose(B), opB=transpose(A)
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // measure GPU time with events
    cudaEvent_t start_ev, stop_ev;
    CHECK_CUDA(cudaEventCreate(&start_ev));
    CHECK_CUDA(cudaEventCreate(&stop_ev));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start_ev, 0));

    // note: leading dimension = N (since matrices are N x N)
    CHECK_CUBLAS( cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_C, N
    ) );

    CHECK_CUDA(cudaEventRecord(stop_ev, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_ev));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start_ev, stop_ev));

    // copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sz * sizeof(float), cudaMemcpyDeviceToHost));

    // cleanup cuBLAS and device buffers
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start_ev));
    CHECK_CUDA(cudaEventDestroy(stop_ev));

    // write files (no extension)
    char fnameA[256], fnameB[256], fnameC[256];
    snprintf(fnameA, sizeof(fnameA), "A_%d", N);
    snprintf(fnameB, sizeof(fnameB), "B_%d", N);
    snprintf(fnameC, sizeof(fnameC), "C_%d", N);

    FILE *fa = fopen(fnameA, "wb");
    FILE *fb = fopen(fnameB, "wb");
    FILE *fc = fopen(fnameC, "wb");
    if(!fa || !fb || !fc){
        perror("fopen failed");
        if(fa) fclose(fa);
        if(fb) fclose(fb);
        if(fc) fclose(fc);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }
    if(fwrite(h_A, sizeof(float), sz, fa) != sz) perror("fwrite A");
    if(fwrite(h_B, sizeof(float), sz, fb) != sz) perror("fwrite B");
    if(fwrite(h_C, sizeof(float), sz, fc) != sz) perror("fwrite C");
    fclose(fa); fclose(fb); fclose(fc);

    free(h_A); free(h_B); free(h_C);

    printf("Wrote %s, %s, %s  (binary float32, row-major). GPU time: %.3f ms\n", fnameA, fnameB, fnameC, ms);
    return 0;
}