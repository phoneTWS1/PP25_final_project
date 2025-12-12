NVCC = nvcc
CXX  = g++
CC = gcc

CUDA_SRC := strassen_par.cu
CPP_SRC  := strassen_seq.cc
BLOCK_SRC := block_par.cu

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
CXXFLAGS  := -O3 -std=c++11 -march=native
LDFLAGS = -lm

TARGETS := strassen_par strassen_seq block_par

.PHONY: all clean

all: $(TARGETS)

strassen_par: $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $?

strassen_seq: $(CPP_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

block_par: $(BLOCK_SRC)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<
	
clean:
	rm -f $(TARGETS) *.o