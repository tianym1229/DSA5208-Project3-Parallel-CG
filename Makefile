CXX ?= g++
MPICXX ?= mpicxx
NVCC ?= nvcc

CXXFLAGS_COMMON = -O3 -std=c++17 -Wall -Wextra -pedantic -Iinclude

CXXFLAGS_SERIAL = $(CXXFLAGS_COMMON)
CXXFLAGS_MPI = $(CXXFLAGS_COMMON) -fopenmp
CXXFLAGS_HYBRID = $(CXXFLAGS_COMMON) -fopenmp

NVCCFLAGS = -std=c++17 -O3

BUILD_DIR = build
GPU_EXE = $(BUILD_DIR)/cg_gpu

.PHONY: all clean gpu serial-test mpi-test hybrid-test benchmark-test part2-test

all: cg_serial cg_mpi cg_hybrid

cg_serial: src/main.cpp src/cg.cpp include/cg.hpp
	$(CXX) $(CXXFLAGS_SERIAL) src/main.cpp src/cg.cpp -o $@

cg_mpi: src/mpi_main.cpp include/cg.hpp
	$(MPICXX) $(CXXFLAGS_MPI) src/mpi_main.cpp -o $@

cg_hybrid: src/mpi_main.cpp include/cg.hpp
	$(MPICXX) $(CXXFLAGS_HYBRID) src/mpi_main.cpp -o $@

gpu: $(GPU_EXE)

$(GPU_EXE): src/gpu/gpu_main.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -ccbin /usr/bin/g++ \
		src/gpu/gpu_main.cu -lcublas -lcusparse -o $@

serial-test: cg_serial
	python3 tests/test_serial.py

mpi-test: cg_mpi cg_serial
	python3 tests/test_mpi.py

hybrid-test: cg_hybrid cg_mpi
	python3 tests/test_hybrid.py

benchmark-test: cg_serial cg_mpi
	python3 tests/test_benchmark.py

part2-test: cg_mpi
	python3 tests/test_part2.py

clean:
	rm -f cg_serial cg_mpi cg_hybrid
	rm -f $(GPU_EXE)
