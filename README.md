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

## Blocked Matrix Multiplication
- `block_mul_gmem.cu` : optimized by shared memory
<pre style="overflow-x:auto; white-space:pre;">
==2939817== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, float*, float*, float*)
        1              gld_transactions_per_request                       Global Load Transactions Per Request   16.000000   16.000000   16.000000
        1                      global_load_requests   Total number of global load requests from Multiprocessor  2148007936  2148007936  2148007936
        1                          gld_transactions                                   Global Load Transactions  8592031746  8592031746  8592031746
        1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)      93.87%      93.87%      93.87%
        1                      dram_read_throughput                              Device Memory Read Throughput  17.018GB/s  17.018GB/s  17.018GB/s
        1                            gld_throughput                                     Global Load Throughput  651.42GB/s  651.42GB/s  651.42GB/s
        1                        achieved_occupancy                                         Achieved Occupancy    0.994573    0.994573    0.994573
        1                             sm_efficiency                                    Multiprocessor Activity      99.69%      99.69%      99.69%


==2938061== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
GPU activities:   85.88%  66.219ms         1  66.219ms  66.219ms  66.219ms  block_mul_kernel(int, int, float*, float*, float*)
                    9.52%  7.3445ms         2  3.6723ms  3.6453ms  3.6992ms  [CUDA memcpy HtoD]
                    4.60%  3.5473ms         1  3.5473ms  3.5473ms  3.5473ms  [CUDA memcpy DtoH]

</pre>
- `block_mul_shmem.cu`:
<pre style="overflow-x:auto; white-space:pre;">
==2940490== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, float*, float*, float*)
        1              gld_transactions_per_request                       Global Load Transactions Per Request   16.000000   16.000000   16.000000
        1                      global_load_requests   Total number of global load requests from Multiprocessor    67108864    67108864    67108864
        1                          gld_transactions                                   Global Load Transactions   268435458   268435458   268435458
        1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       6.65%       6.65%       6.65%
        1                      dram_read_throughput                              Device Memory Read Throughput  13.290GB/s  13.290GB/s  13.290GB/s
        1                            gld_throughput                                     Global Load Throughput  202.86GB/s  202.86GB/s  202.86GB/s
        1                        achieved_occupancy                                         Achieved Occupancy    0.998754    0.998754    0.998754
        1                             sm_efficiency                                    Multiprocessor Activity      99.78%      99.78%      99.78%


==2937636== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
GPU activities:   58.20%  14.895ms         1  14.895ms  14.895ms  14.895ms  block_mul_kernel(int, int, float*, float*, float*)
                27.88%  7.1347ms         2  3.5674ms  3.5092ms  3.6255ms  [CUDA memcpy HtoD]
                13.92%  3.5626ms         1  3.5626ms  3.5626ms  3.5626ms  [CUDA memcpy DtoH]
</pre>
- `block_mul_coal.cu`:
<pre style="overflow-x:auto; white-space:pre;">
==2940530== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, float*, float*, float*)
        1              gld_transactions_per_request                       Global Load Transactions Per Request   16.000000   16.000000   16.000000
        1                      global_load_requests   Total number of global load requests from Multiprocessor    67108864    67108864    67108864
        1                          gld_transactions                                   Global Load Transactions   268435458   268435458   268435458
        1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)      10.89%      10.89%      10.89%
        1                      dram_read_throughput                              Device Memory Read Throughput  76.591GB/s  76.591GB/s  76.591GB/s
        1                            gld_throughput                                     Global Load Throughput  148.63GB/s  148.63GB/s  148.63GB/s
        1                        achieved_occupancy                                         Achieved Occupancy    0.997830    0.997830    0.997830
        1                             sm_efficiency                                    Multiprocessor Activity      99.62%      99.62%      99.62%

==2941318== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
GPU activities:   57.94%  14.855ms         1  14.855ms  14.855ms  14.855ms  block_mul_kernel(int, int, float*, float*, float*)
                28.19%  7.2288ms         2  3.6144ms  3.6061ms  3.6227ms  [CUDA memcpy HtoD]
</pre>

In the kernel, global memory is only accessed when loading a new A/B tile; all the heavy work inside the inner k-loop uses shared memory and registers. That means most of the runtime is spent doing arithmetic and shared-mem access rather than repeatedly hitting global memory, 

Therefore kernel already has decent arithmetic intensity (a lot of FLOPs per global load). In this situation, improving coalescing on those tile loads is still good and can give some speedup, but it won’t transform performance by 10× because global memory isn’t the dominant cost anymore. 

The real performance gain is from reusing shared mmemory. Each A/B element is loaded once from global but reused many times from shared memory, and that reuse is what primarily accelerates the code compared to a naive pure-global-memory version.
- `block_mul_bank.cu`:
