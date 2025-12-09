# PP25_final_project

## Test case
`N` is a input argument `N = atoi(argv[1])`  
```
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
```    
## Correctness check
```
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
```