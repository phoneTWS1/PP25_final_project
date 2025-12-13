#!/usr/bin/env bash
set -euo pipefail

EXE=./exe/double/8w_vec
OUT=8w_vec

srun -p nvidia -n1 --gres=gpu:1 \
    nvprof --metrics gld_transactions,global_load_requests,gst_transactions,global_store_request,dram_read_throughput,stall_memory_dependency,stall_exec_dependency,achieved_occupancy,sm_efficiency,shared_load_transactions,shared_load_throughput,shared_store_transactions,shared_store_throughput,shared_efficiency,shared_transaction_per_request,shared_utilization,inst_fp_64,flop_count_dp,flop_dp_efficiency \
    $EXE 4096 ./dataset/A_4096 ./dataset/B_4096 ./dataset/C_4096 \
    2> ./block_mul/profile/$OUT/metric_4096.txt