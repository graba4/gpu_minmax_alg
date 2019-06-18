# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-10.1 #change if you need to
COMPILE := nvcc
#/usr/local/cuda-8.0
#/opt/cuda-8.0
ARCH := sm_30

# Common includes and paths for CUDA
INCLUDES  := -Icommon/inc
HOST_COMPILER ?= gcc
HOST_COMPILER_ARGS := -pedantic -O3 -Wall -std=c99 -D_XOPEN_SOURCE=500 -D_BSD_SOURCE -g 

NVCC = $(COMPILE) $(CCFLAGS)
CCFLAGS = -Wno-deprecated-gpu-targets -O3 -D_FORCE_INLINES --gpu-architecture=$(ARCH)
C_COMPILE = -ccbin $(HOST_COMPILER) $(addprefix -Xcompiler , $(HOST_COMPILER_ARGS))

WIDTH = 200

all: build check

build: min_max

min_max.o:min_max.cpp
	$(NVCC) $(INCLUDES) -o $@ -c $<

min_max: min_max.o resources.o io.o errors.o cuda_deque.o implementations.o implementations_cpu.o
	$(NVCC) -o $@ $+ 

io.o: io.cpp
	$(NVCC) $(INCLUDES) -o $@ -c $<

errors.o: errors.cpp
	$(NVCC) $(INCLUDES) -o $@ -c $<

cuda_deque.o: cuda_deque.cu
	$(NVCC) $(INCLUDES) -dc -o $@ -c $<

implementations.o: implementations.cu
	$(NVCC) $(INCLUDES) -dc -o $@ -c $<

implementations_cpu.o: implementations_cpu.cu
	$(NVCC) $(INCLUDES) -o $@ -c $<

resources.o: resources.cu
	$(NVCC) $(INCLUDES) -o $@ -c $<

check:
	./min_max -v 15000000 -w 3 -c 1 -i 2 -t 10 -r 2 -a

clean:
	rm -f min_max *.o
