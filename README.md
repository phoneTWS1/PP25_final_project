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
        A[i] = (float)i;
        B[i] = 0.0f;
        C[i] = 0.0f;
    }

    for(int i=0 ; i<N ; i++){
        B[i * N + i] = 1.0f;
    }
    
}
```    
## Correctness check
```
void correctness_check(){
    int cnt = 0;
    for(int i = 0 ; i < N * N ; i++){
        if(C[i] != A[i])
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

## Global memory profiling
use `nvprof` metrics. GTX1080 doesn't support `ncu` (nsight compute)

### coalesing 
- `gld_transactions_per_request`:  Average number of global memory load transactions performed for each global memory load.
- `global_load_requests`: Total number of global load requests from Multiprocessor
- `gld_transactions`: Number of global memory load transaction
- `stall_memory_dependency` : Percentage of stalls occurring because a memory operation cannot be performed due to the required resources not being available or fully utilized, or because too many requests of a given type are outstanding`
- `dram_read_throughput`: Device memory read throughput
- `gld_throughput`: Global memory load throughput
   
If coalesing  
1. `gld_transactions_per_request` should close to 1
2. `gld_load_request` ~ `gld_transaction`
3. `stall_memory_dependency` shoudl be small
4. `dram_read_througput` should close to peak 320GB/s on GTX 1080
5. `gld_througtput` is just a theratical throughput of our code.

### wrap level
- `achieved_occupancy`:  Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor
- `sm_efficiency`: The percentage of time at least one warp is active on a specific multiprocessor
