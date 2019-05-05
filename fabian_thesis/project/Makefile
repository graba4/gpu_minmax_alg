# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-8.0
#/usr/local/cuda-8.0
#/opt/cuda-8.0
ARCH := sm_30

# Common includes and paths for CUDA
INCLUDES  := -Icommon/inc
HOST_COMPILER ?= gcc
HOST_COMPILER_ARGS := -pedantic -O3 -Wall -std=c99 -D_XOPEN_SOURCE=500 -D_BSD_SOURCE -g 

NVCC = $(CUDA_PATH)/bin/nvcc $(CCFLAGS)
CCFLAGS = -Wno-deprecated-gpu-targets -O3 --gpu-architecture=$(ARCH)
C_COMPILE = -ccbin $(HOST_COMPILER) $(addprefix -Xcompiler , $(HOST_COMPILER_ARGS))

WIDTH = 3800

all: build check

build: lin-sys-solv

lin-sys-solv.o:lin-sys-solv.cpp
	$(NVCC) $(INCLUDES) -o $@ -c $<

lin-sys-solv: implementations.o lin-sys-solv.o io.o errors.o resources.o implementations_cpu.o
	$(NVCC) -o $@ $+ 

io.o: io.cpp
	$(NVCC) $(INCLUDES) -o $@ -c $<

errors.o: errors.cpp
	$(NVCC) $(INCLUDES) -o $@ -c $<

implementations.o: implementations.cu
	$(NVCC) $(INCLUDES) -o $@ -c $<

implementations_cpu.o: implementations_cpu.cpp
	$(NVCC) $(INCLUDES) -o $@ -c $<

resources.o: resources.cu
	$(NVCC) $(INCLUDES) -o $@ -c $<

check:
	./lin-sys-solv -v $(WIDTH) -c 4 -i 4 -r 5 -t 1300
	./lin-sys-solv -v $(WIDTH) -c 4 -i 1 -r 5 -t 1300

clean:
	rm -f lin-sys-solv *.o
