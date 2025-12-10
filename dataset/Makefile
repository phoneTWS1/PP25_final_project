CXX = g++
CXXFLAGS := -O3 -std=c++17 -march=native
SRC := generator.cc
TARGET := generator

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	@if [ -z "$(N)" ]; then echo "請提供 N，例如: make run N=1024"; exit 1; fi
	./$(TARGET) $(N)

clean:
	-rm -f $(TARGET)