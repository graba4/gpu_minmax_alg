#ifndef LOCK_H
#define LOCK_H
#include <cuda_runtime.h>

typedef struct Lock
{
	int	*mutex;

	Lock(){
		int state=0;
		checkCudaErrors(cudaMalloc((void**)&mutex,sizeof(int)));
		checkCudaErrors(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
	}

	~Lock(){
		cudaFree(mutex);
	}

	__device__ void lock(){
		while(atomicCAS(mutex,0,1)!=0);
	}
	
	__device__ void unlock(){
		atomicExch(mutex,0);
	}
}Lock;

#endif