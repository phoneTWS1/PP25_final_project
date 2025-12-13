#!/usr/bin/env bash
set -euo pipefail

EXE=./exe/double/4w_vec
OUT=4w_vec

mkdir -p ./block_mul/profile/$OUT

srun -p nvidia -n1 --gres=gpu:1 nvprof $EXE 64 ./dataset/A_64 ./dataset/B_64 ./dataset/C_64 2> ./block_mul/profile/$OUT/64.txt
srun -p nvidia -n1 --gres=gpu:1 nvprof $EXE 128 ./dataset/A_128 ./dataset/B_128 ./dataset/C_128 2> ./block_mul/profile/$OUT/128.txt
srun -p nvidia -n1 --gres=gpu:1 nvprof $EXE 256 ./dataset/A_256 ./dataset/B_256 ./dataset/C_256 2> ./block_mul/profile/$OUT/256.txt
srun -p nvidia -n1 --gres=gpu:1 nvprof $EXE 512 ./dataset/A_512 ./dataset/B_512 ./dataset/C_512 2> ./block_mul/profile/$OUT/512.txt
srun -p nvidia -n1 --gres=gpu:1 nvprof $EXE 1024 ./dataset/A_1024 ./dataset/B_1024 ./dataset/C_1024 2> ./block_mul/profile/$OUT/1024.txt
srun -p nvidia -n1 --gres=gpu:1 nvprof $EXE 2048 ./dataset/A_2048 ./dataset/B_2048 ./dataset/C_2048 2> ./block_mul/profile/$OUT/2048.txt
srun -p nvidia -n1 --gres=gpu:1 nvprof $EXE 4096 ./dataset/A_4096 ./dataset/B_4096 ./dataset/C_4096 2> ./block_mul/profile/$OUT/4096.txt
srun -p nvidia -n1 --gres=gpu:1 nvprof $EXE 8192 ./dataset/A_8192 ./dataset/B_8192 ./dataset/C_8192 2> ./block_mul/profile/$OUT/8192.txt