NVCC = nvcc
CXX  = g++

CUDA_SRC := strassen_par.cu
CPP_SRC  := strassen_seq.cc

NVCCFLAGS := -O3 -arch=sm_60
CXXFLAGS  := -O3 -std=c++17 -march=native

TARGETS := strassen_par strassen_seq

.PHONY: all clean

all: $(TARGETS)

strassen_par: $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

strassen_seq: $(CPP_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGETS) *.o