NVCC = nvcc
CXX  = g++

CUDA_SRC := block_mul_par.cu
CPP_SRC  := block_mul_seq.cc

NVCCFLAGS := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
CXXFLAGS  := -O3 -std=c++17 -march=native

TARGETS := block_mul_par block_mul_seq

.PHONY: all clean

all: $(TARGETS)

block_mul_par: $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

block_mul_seq: $(CPP_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGETS) *.o