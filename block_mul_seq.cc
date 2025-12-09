#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int N;
float *A, *B;
float *C;

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

int main(int argc, char* argv[]){
    assert(argc==2);
    N = atoi(argv[1]);
    create_matrix();
 
    int T = 32; // tile size
    int n = N / T;
    assert(N % T ==0);

     // c[ii, jj] = A[ii, kk] B[kk, jj] for all kk
    for(int ii=0; ii<n; ii++){
        for(int jj=0; jj<n; jj++){
            for(int kk=0; kk<n ;kk++){

                // c[ii,jj][i,j] = A[ii, kk][i,k] B[kk, jj][k,j] for all kk
                for(int i=0; i < T; i++ ){
                    for(int j=0; j < T; j++){
                        for(int k=0; k<T; k++){
                            C[(ii*T + i)*N + (jj*T+j)] += (A[(ii*T+i)*N + (kk*T+k)] * B[(kk*T+k)*N + (jj*T+j)]);
                        }
                    }
                    
                }
  
            }
        }
    }

    correctness_check();

    
    return 0;
}

