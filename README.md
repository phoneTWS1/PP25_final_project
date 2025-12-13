# PP25_final_project

- `GFLOP/S`
- `global_load_requests`: Total number of global load requests from Multiprocessor
- `gld_transactions`: Number of global memory load transaction
- `stall_memory_dependency` : Percentage of stalls occurring because a memory operation cannot be performed due to the required resources not being available or fully utilized, or because too many requests of a given type are outstanding`
- `dram_read_throughput`: Device memory read throughput

All the version's tile size is  32, n = N/32
# Global Memeory version
This is the baseline block matrix multiplication baseline. kernel is lauch with `<<<dim3(n,n), dim3(32,32)>>>`, a warp is responsible for a row of a cile. A thread is responsible for a elememt in a row. globe memory acess pattern is as folloing, which is load A by broadcase in a wrap and coalesced memory access loading B.

<pre style="overflow-x:auto; white-space:pre;">
==3129073== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  1.3746e+11  1.3746e+11  1.3746e+11
          1                      global_load_requests   Total number of global load requests from Multiprocessor  1.7182e+10  1.7182e+10  1.7182e+10
          1                          gst_transactions                                  Global Store Transactions  1.7180e+10  1.7180e+10  1.7180e+10
          1                      dram_read_throughput                              Device Memory Read Throughput  22.296GB/s  22.296GB/s  22.296GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)      90.75%      90.75%      90.75%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)       8.35%       8.35%       8.35%
          1                        achieved_occupancy                                         Achieved Occupancy    0.998377    0.998377    0.998377
          1                             sm_efficiency                                    Multiprocessor Activity      99.92%      99.92%      99.92%
          1                  shared_load_transactions                                   Shared Load Transactions           0           0           0
          1                    shared_load_throughput                              Shared Memory Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                 shared_store_transactions                                  Shared Store Transactions           0           0           0
          1                   shared_store_throughput                             Shared Memory Store Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                         shared_efficiency                                   Shared Memory Efficiency       0.00%       0.00%       0.00%
          1                        shared_utilization                                  Shared Memory Utilization    Idle (0)    Idle (0)    Idle (0)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      54.92%      54.92%      54.92%

==3110987== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   87.46%  876.66ms         1  876.66ms  876.66ms  876.66ms  block_mul_kernel(int, int, double*, double*, double*)
                    6.64%  66.607ms         1  66.607ms  66.607ms  66.607ms  [CUDA memcpy DtoH]
                    5.90%  59.109ms         2  29.555ms  29.252ms  29.857ms  [CUDA memcpy HtoD]
</pre>
1. `gld_transactions / global_load_requests ≈ 8` matches our access pattern. For each warp loading a B-tile row, 32 threads load 32 doubles = 256 bytes, which requires 256 / 32 = 8 transactions. The A-tile load is a broadcast and can be served by about 1 transaction per warp. 
2. The relatively low DRAM read throughput (compared to peak bandwidth), together with the very high stall_memory_dependency, implies that many load requests are waiting on memory rather than saturating bandwidth. This is consistent with a memory-latency-bound kernel.
3. Although achieved occupancy and SM efficiency are high, most warps are “busy” just waiting for global memory loads to complete, not doing arithmetic
This shows that with the global-only version, the bottleneck is global memory access, not compute.

# Shared memory version
For share memory version we compare two version one with coalsing memory access and one with non_coalsesing memoru access
## Non coalesing
<pre style="overflow-x:auto; white-space:pre;">
==3129114== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  4294967298  4294967298  4294967298
          1                      global_load_requests   Total number of global load requests from Multiprocessor   536870912   536870912   536870912
          1                          gst_transactions                                  Global Store Transactions    16777216    16777216    16777216
          1                      dram_read_throughput                              Device Memory Read Throughput  11.978GB/s  11.978GB/s  11.978GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       3.20%       3.20%       3.20%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      40.13%      40.13%      40.13%
          1                        achieved_occupancy                                         Achieved Occupancy    0.999591    0.999591    0.999591
          1                             sm_efficiency                                    Multiprocessor Activity      99.92%      99.92%      99.92%
          1                  shared_load_transactions                                   Shared Load Transactions  3.6507e+10  3.6507e+10  3.6507e+10
          1                    shared_load_throughput                              Shared Memory Load Throughput  2779.5GB/s  2779.5GB/s  2779.5GB/s
          1                 shared_store_transactions                                  Shared Store Transactions  4294967296  4294967296  4294967296
          1                   shared_store_throughput                             Shared Memory Store Throughput  327.00GB/s  327.00GB/s  327.00GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      11.51%      11.51%      11.51%
          1                        shared_utilization                                  Shared Memory Utilization    High (7)    High (7)    High (7)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      28.57%      28.57%      28.57%

==2181603== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   93.26%  1.62920s         1  1.62920s  1.62920s  1.62920s  block_mul_kernel(int, int, double*, double*, double*)
                    3.39%  59.268ms         2  29.634ms  29.600ms  29.668ms  [CUDA memcpy HtoD]
                    3.34%  58.401ms         1  58.401ms  58.401ms  58.401ms  [CUDA memcpy DtoH]
</pre>

## Coalesing
<pre style="overflow-x:auto; white-space:pre;">
==3129157== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  4294967298  4294967298  4294967298
          1                      global_load_requests   Total number of global load requests from Multiprocessor   536870912   536870912   536870912
          1                          gst_transactions                                  Global Store Transactions     4194304     4194304     4194304
          1                      dram_read_throughput                              Device Memory Read Throughput  29.197GB/s  29.197GB/s  29.197GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       2.27%       2.27%       2.27%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      41.18%      41.18%      41.18%
          1                        achieved_occupancy                                         Achieved Occupancy    0.999626    0.999626    0.999626
          1                             sm_efficiency                                    Multiprocessor Activity      99.88%      99.88%      99.88%
          1                  shared_load_transactions                                   Shared Load Transactions  6442450944  6442450944  6442450944
          1                    shared_load_throughput                              Shared Memory Load Throughput  1135.0GB/s  1135.0GB/s  1135.0GB/s
          1                 shared_store_transactions                                  Shared Store Transactions   268435456   268435456   268435456
          1                   shared_store_throughput                             Shared Memory Store Throughput  47.292GB/s  47.292GB/s  47.292GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      70.00%      70.00%      70.00%
          1                        shared_utilization                                  Shared Memory Utilization     Low (3)     Low (3)     Low (3)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      67.32%      67.32%      67.32%

==3110987== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   87.46%  876.66ms         1  876.66ms  876.66ms  876.66ms  block_mul_kernel(int, int, double*, double*, double*)
                    6.64%  66.607ms         1  66.607ms  66.607ms  66.607ms  [CUDA memcpy DtoH]
                    5.90%  59.109ms         2  29.555ms  29.252ms  29.857ms  [CUDA memcpy HtoD]
</pre>

1. Compare to global memory version, both implementation have massively reduced the number of global memory load requests and transaction. 
# Resisteing reusing
## 8 warp
<pre style="overflow-x:auto; white-space:pre;">
==3129196== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  4294967298  4294967298  4294967298
          1                      global_load_requests   Total number of global load requests from Multiprocessor   536870912   536870912   536870912
          1                          gst_transactions                                  Global Store Transactions     4194304     4194304     4194304
          1                      dram_read_throughput                              Device Memory Read Throughput  35.414GB/s  35.414GB/s  35.414GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       0.65%       0.65%       0.65%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      52.37%      52.37%      52.37%
          1                        achieved_occupancy                                         Achieved Occupancy    0.749123    0.749123    0.749123
          1                             sm_efficiency                                    Multiprocessor Activity      99.84%      99.84%      99.84%
          1                  shared_load_transactions                                   Shared Load Transactions  3221225472  3221225472  3221225472
          1                    shared_load_throughput                              Shared Memory Load Throughput  671.27GB/s  671.27GB/s  671.27GB/s
          1                 shared_store_transactions                                  Shared Store Transactions   268435456   268435456   268435456
          1                   shared_store_throughput                             Shared Memory Store Throughput  55.939GB/s  55.939GB/s  55.939GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      42.31%      42.31%      42.31%
          1                        shared_utilization                                  Shared Memory Utilization     Low (2)     Low (2)     Low (2)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      78.93%      78.93%      78.93%

==3111517== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   82.29%  582.79ms         1  582.79ms  582.79ms  582.79ms  block_mul_kernel(int, int, double*, double*, double*)
                    9.33%  66.044ms         1  66.044ms  66.044ms  66.044ms  [CUDA memcpy DtoH]
                    8.39%  59.389ms         2  29.694ms  29.665ms  29.724ms  [CUDA memcpy HtoD]
</pre>
## 4 warp
<pre style="overflow-x:auto; white-space:pre;">
==3129237== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  4294967298  4294967298  4294967298
          1                      global_load_requests   Total number of global load requests from Multiprocessor   536870912   536870912   536870912
          1                          gst_transactions                                  Global Store Transactions     4194304     4194304     4194304
          1                      dram_read_throughput                              Device Memory Read Throughput  29.204GB/s  29.204GB/s  29.204GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       0.94%       0.94%       0.94%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      86.56%      86.56%      86.56%
          1                        achieved_occupancy                                         Achieved Occupancy    0.374906    0.374906    0.374906
          1                             sm_efficiency                                    Multiprocessor Activity      99.69%      99.69%      99.69%
          1                  shared_load_transactions                                   Shared Load Transactions  2684354560  2684354560  2684354560
          1                    shared_load_throughput                              Shared Memory Load Throughput  572.23GB/s  572.23GB/s  572.23GB/s
          1                 shared_store_transactions                                  Shared Store Transactions   268435456   268435456   268435456
          1                   shared_store_throughput                             Shared Memory Store Throughput  57.223GB/s  57.223GB/s  57.223GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      31.82%      31.82%      31.82%
          1                        shared_utilization                                  Shared Memory Utilization     Low (2)     Low (2)     Low (2)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      81.24%      81.24%      81.24%

==3112737== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   81.49%  564.57ms         1  564.57ms  564.57ms  564.57ms  block_mul_kernel(int, int, double*, double*, double*)
                    9.74%  67.463ms         1  67.463ms  67.463ms  67.463ms  [CUDA memcpy DtoH]
                    8.77%  60.742ms         2  30.371ms  30.193ms  30.549ms  [CUDA memcpy HtoD]
</pre>
## 2 warp
<pre style="overflow-x:auto; white-space:pre;">
==3129282== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  4294967298  4294967298  4294967298
          1                      global_load_requests   Total number of global load requests from Multiprocessor   536870912   536870912   536870912
          1                          gst_transactions                                  Global Store Transactions     4194304     4194304     4194304
          1                      dram_read_throughput                              Device Memory Read Throughput  30.139GB/s  30.139GB/s  30.139GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       1.68%       1.68%       1.68%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      95.60%      95.60%      95.60%
          1                        achieved_occupancy                                         Achieved Occupancy    0.187456    0.187456    0.187456
          1                             sm_efficiency                                    Multiprocessor Activity      99.67%      99.67%      99.67%
          1                  shared_load_transactions                                   Shared Load Transactions  2415919104  2415919104  2415919104
          1                    shared_load_throughput                              Shared Memory Load Throughput  530.40GB/s  530.40GB/s  530.40GB/s
          1                 shared_store_transactions                                  Shared Store Transactions   268435456   268435456   268435456
          1                   shared_store_throughput                             Shared Memory Store Throughput  58.933GB/s  58.933GB/s  58.933GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      25.00%      25.00%      25.00%
          1                        shared_utilization                                  Shared Memory Utilization     Low (2)     Low (2)     Low (2)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      82.48%      82.48%      82.48%

==3113340== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   81.00%  537.02ms         1  537.02ms  537.02ms  537.02ms  block_mul_kernel(int, int, double*, double*, double*)
                   10.03%  66.501ms         1  66.501ms  66.501ms  66.501ms  [CUDA memcpy DtoH]
                    8.97%  59.467ms         2  29.733ms  29.691ms  29.776ms  [CUDA memcpy HtoD]
</pre>

# Better mapping
## 8 warp
<pre style="overflow-x:auto; white-space:pre;">
==3129321== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  4294967298  4294967298  4294967298
          1                      global_load_requests   Total number of global load requests from Multiprocessor   536870912   536870912   536870912
          1                          gst_transactions                                  Global Store Transactions     4194304     4194304     4194304
          1                      dram_read_throughput                              Device Memory Read Throughput  29.058GB/s  29.058GB/s  29.058GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       1.82%       1.82%       1.82%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      76.34%      76.34%      76.34%
          1                        achieved_occupancy                                         Achieved Occupancy    0.499867    0.499867    0.499867
          1                             sm_efficiency                                    Multiprocessor Activity      99.90%      99.90%      99.90%
          1                  shared_load_transactions                                   Shared Load Transactions  3221225472  3221225472  3221225472
          1                    shared_load_throughput                              Shared Memory Load Throughput  674.02GB/s  674.02GB/s  674.02GB/s
          1                 shared_store_transactions                                  Shared Store Transactions   268435456   268435456   268435456
          1                   shared_store_throughput                             Shared Memory Store Throughput  56.169GB/s  56.169GB/s  56.169GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      42.31%      42.31%      42.31%
          1                        shared_utilization                                  Shared Memory Utilization     Low (2)     Low (2)     Low (2)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      79.13%      79.13%      79.13%

==3111661== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   81.67%  564.84ms         1  564.84ms  564.84ms  564.84ms  block_mul_kernel(int, int, double*, double*, double*)
                    9.71%  67.135ms         1  67.135ms  67.135ms  67.135ms  [CUDA memcpy DtoH]
                    8.62%  59.608ms         2  29.804ms  29.751ms  29.858ms  [CUDA memcpy HtoD]
</pre>
## 4 warp
<pre style="overflow-x:auto; white-space:pre;">
==3129362== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  4294967298  4294967298  4294967298
          1                      global_load_requests   Total number of global load requests from Multiprocessor   536870912   536870912   536870912
          1                          gst_transactions                                  Global Store Transactions     4194304     4194304     4194304
          1                      dram_read_throughput                              Device Memory Read Throughput  30.616GB/s  30.616GB/s  30.616GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       1.22%       1.22%       1.22%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      82.13%      82.13%      82.13%
          1                        achieved_occupancy                                         Achieved Occupancy    0.374864    0.374864    0.374864
          1                             sm_efficiency                                    Multiprocessor Activity      99.68%      99.68%      99.68%
          1                  shared_load_transactions                                   Shared Load Transactions  2147483648  2147483648  2147483648
          1                    shared_load_throughput                              Shared Memory Load Throughput  480.25GB/s  480.25GB/s  480.25GB/s
          1                 shared_store_transactions                                  Shared Store Transactions   268435456   268435456   268435456
          1                   shared_store_throughput                             Shared Memory Store Throughput  60.031GB/s  60.031GB/s  60.031GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      38.89%      38.89%      38.89%
          1                        shared_utilization                                  Shared Memory Utilization     Low (2)     Low (2)     Low (2)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      83.93%      83.93%      83.93%

==3113196== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   80.56%  543.68ms         1  543.68ms  543.68ms  543.68ms  block_mul_kernel(int, int, double*, double*, double*)
                   10.41%  70.248ms         1  70.248ms  70.248ms  70.248ms  [CUDA memcpy DtoH]
                    9.04%  60.980ms         2  30.490ms  29.778ms  31.202ms  [CUDA memcpy HtoD]
</pre>
## 2 warp
<pre style="overflow-x:auto; white-space:pre;">
==3129402== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  4294967298  4294967298  4294967298
          1                      global_load_requests   Total number of global load requests from Multiprocessor   536870912   536870912   536870912
          1                          gst_transactions                                  Global Store Transactions     4194304     4194304     4194304
          1                      dram_read_throughput                              Device Memory Read Throughput  33.464GB/s  33.464GB/s  33.464GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       2.94%       2.94%       2.94%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      93.56%      93.56%      93.56%
          1                        achieved_occupancy                                         Achieved Occupancy    0.187350    0.187350    0.187350
          1                             sm_efficiency                                    Multiprocessor Activity      99.75%      99.75%      99.75%
          1                  shared_load_transactions                                   Shared Load Transactions  2147483648  2147483648  2147483648
          1                    shared_load_throughput                              Shared Memory Load Throughput  475.17GB/s  475.17GB/s  475.17GB/s
          1                 shared_store_transactions                                  Shared Store Transactions   536870912   536870912   536870912
          1                   shared_store_throughput                             Shared Memory Store Throughput  118.79GB/s  118.79GB/s  118.79GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      25.00%      25.00%      25.00%
          1                        shared_utilization                                  Shared Memory Utilization     Low (2)     Low (2)     Low (2)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      83.41%      83.41%      83.41%

==3113723== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   81.11%  549.68ms         1  549.68ms  549.68ms  549.68ms  block_mul_kernel(int, int, double*, double*, double*)
                   10.19%  69.057ms         1  69.057ms  69.057ms  69.057ms  [CUDA memcpy DtoH]
                    8.70%  58.979ms         2  29.490ms  29.486ms  29.494ms  [CUDA memcpy HtoD]
</pre>

# Vectorization
## 8 warp
<pre style="overflow-x:auto; white-space:pre;">
==3129527== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  8589934594  8589934594  8589934594
          1                      global_load_requests   Total number of global load requests from Multiprocessor   536870912   536870912   536870912
          1                          gst_transactions                                  Global Store Transactions     4194304     4194304     4194304
          1                      dram_read_throughput                              Device Memory Read Throughput  32.735GB/s  32.735GB/s  32.735GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       1.44%       1.44%       1.44%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      67.19%      67.19%      67.19%
          1                        achieved_occupancy                                         Achieved Occupancy    0.624398    0.624398    0.624398
          1                             sm_efficiency                                    Multiprocessor Activity      99.90%      99.90%      99.90%
          1                  shared_load_transactions                                   Shared Load Transactions  3221225472  3221225472  3221225472
          1                    shared_load_throughput                              Shared Memory Load Throughput  644.37GB/s  644.37GB/s  644.37GB/s
          1                 shared_store_transactions                                  Shared Store Transactions  1342177280  1342177280  1342177280
          1                   shared_store_throughput                             Shared Memory Store Throughput  268.49GB/s  268.49GB/s  268.49GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      32.35%      32.35%      32.35%
          1                        shared_utilization                                  Shared Memory Utilization     Low (2)     Low (2)     Low (2)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      75.13%      75.13%      75.13%

==3111805== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   82.58%  590.23ms         1  590.23ms  590.23ms  590.23ms  block_mul_kernel(int, int, double*, double*, double*)
                    9.11%  65.124ms         1  65.124ms  65.124ms  65.124ms  [CUDA memcpy DtoH]
                    8.30%  59.349ms         2  29.674ms  29.570ms  29.779ms  [CUDA memcpy HtoD]
</pre>
## 4 warp
<pre style="overflow-x:auto; white-space:pre;">
==3129485== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  4294967298  4294967298  4294967298
          1                      global_load_requests   Total number of global load requests from Multiprocessor   268435456   268435456   268435456
          1                          gst_transactions                                  Global Store Transactions     4194304     4194304     4194304
          1                      dram_read_throughput                              Device Memory Read Throughput  30.695GB/s  30.695GB/s  30.695GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       3.24%       3.24%       3.24%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      68.83%      68.83%      68.83%
          1                        achieved_occupancy                                         Achieved Occupancy    0.374883    0.374883    0.374883
          1                             sm_efficiency                                    Multiprocessor Activity      99.84%      99.84%      99.84%
          1                  shared_load_transactions                                   Shared Load Transactions  2147483648  2147483648  2147483648
          1                    shared_load_throughput                              Shared Memory Load Throughput  473.27GB/s  473.27GB/s  473.27GB/s
          1                 shared_store_transactions                                  Shared Store Transactions   268435456   268435456   268435456
          1                   shared_store_throughput                             Shared Memory Store Throughput  59.159GB/s  59.159GB/s  59.159GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      38.89%      38.89%      38.89%
          1                        shared_utilization                                  Shared Memory Utilization     Low (2)     Low (2)     Low (2)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      83.96%      83.96%      83.96%

==3115897== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   80.79%  551.30ms         1  551.30ms  551.30ms  551.30ms  block_mul_kernel(int, int, double*, double*, double*)
                   10.49%  71.587ms         1  71.587ms  71.587ms  71.587ms  [CUDA memcpy DtoH]
                    8.72%  59.529ms         2  29.764ms  29.676ms  29.853ms  [CUDA memcpy HtoD]
</pre>
## 2 warp
<pre style="overflow-x:auto; white-space:pre;">
==3129442== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1080 (0)"
    Kernel: block_mul_kernel(int, int, double*, double*, double*)
          1                          gld_transactions                                   Global Load Transactions  4294967298  4294967298  4294967298
          1                      global_load_requests   Total number of global load requests from Multiprocessor   268435456   268435456   268435456
          1                          gst_transactions                                  Global Store Transactions     4194304     4194304     4194304
          1                      dram_read_throughput                              Device Memory Read Throughput  35.601GB/s  35.601GB/s  35.601GB/s
          1                   stall_memory_dependency                         Issue Stall Reasons (Data Request)       6.64%       6.64%       6.64%
          1                     stall_exec_dependency                 Issue Stall Reasons (Execution Dependency)      90.09%      90.09%      90.09%
          1                        achieved_occupancy                                         Achieved Occupancy    0.187390    0.187390    0.187390
          1                             sm_efficiency                                    Multiprocessor Activity      99.76%      99.76%      99.76%
          1                  shared_load_transactions                                   Shared Load Transactions  2147483648  2147483648  2147483648
          1                    shared_load_throughput                              Shared Memory Load Throughput  479.43GB/s  479.43GB/s  479.43GB/s
          1                 shared_store_transactions                                  Shared Store Transactions   268435456   268435456   268435456
          1                   shared_store_throughput                             Shared Memory Store Throughput  59.929GB/s  59.929GB/s  59.929GB/s
          1                         shared_efficiency                                   Shared Memory Efficiency      27.78%      27.78%      27.78%
          1                        shared_utilization                                  Shared Memory Utilization     Low (2)     Low (2)     Low (2)
          1                                inst_fp_64                                    FP Instructions(Double)  6.8719e+10  6.8719e+10  6.8719e+10
          1                             flop_count_dp                Floating Point Operations(Double Precision)  1.3744e+11  1.3744e+11  1.3744e+11
          1                        flop_dp_efficiency                               FLOP Efficiency(Peak Double)      84.45%      84.45%      84.45%

==3115618== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   81.71%  522.03ms         1  522.03ms  522.03ms  522.03ms  block_mul_kernel(int, int, double*, double*, double*)
                    9.32%  59.541ms         2  29.770ms  29.734ms  29.806ms  [CUDA memcpy HtoD]
                    8.97%  57.309ms         1  57.309ms  57.309ms  57.309ms  [CUDA memcpy DtoH]
</pre>


