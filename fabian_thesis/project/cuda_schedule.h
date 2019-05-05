#ifndef CUDA_SCHEDULE_H
#define CUDA_SCHEDULE_H

#include <helper_cuda.h>

typedef struct Semaphore{
	int *mutex; 
	int n;

	Semaphore(){
		int i = 0;
		n=i;
		int* state = new int(i);
		for(int j=0; j<i; j++)
			state[j] = 0;
		cudaMalloc( (void **)&mutex, sizeof(int)*i );
		cudaMemcpy( mutex, state, sizeof(int)*i, cudaMemcpyHostToDevice);
	}

	void Free( void ){
		for(int j=0; j< n; j++){
			cudaFree(&mutex[j]);
		}
		cudaFree(mutex);
	}

	~Semaphore( void ){
	}

	__device__ void wait( int i ){
		while( atomicCAS( &mutex[i], 0, 1 ) != 0 );
	}

	__device__ void post( int i ){
		atomicExch( &mutex[i], 0);
	}

}Semaphore;

typedef struct Lock
{
	int	*mutex;

	Lock(){
		int state=1;
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

	__device__ bool is_free(){
		return atomicAnd(mutex, 1);
	}
}Lock;
#endif