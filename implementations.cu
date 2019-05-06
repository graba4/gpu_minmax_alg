#include "cuda_deque.h"
#include "implementations.h"
#include "io.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "resources.h"
#include <timer.h>
#include "errors.h"
#include "cuda_schedule.h"

#define ROUND_UP(N, S) (N%S == 0) ? N/S : N/S+1
#define BETWEEN(value, min, max) (value < max && value > min)
#define DEV_ID (0)

__global__ void naive_aproach_one_thread(cuda_matrix *matrix);

double naive_aproach_fabian(cuda_matrix *matrix){
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks,
		threads = matrix->thread_count,
		max_threads = prop.maxThreadsPerBlock,
		max_sm = prop.multiProcessorCount;

	blocks = matrix->core_count; //(matrix->var_count >= max_threads)? (matrix->var_count/max_threads)+1 : 1;
	/*
	if (matrix->thread_count < 0)
		threads = (matrix->var_count >= max_threads)? max_threads : matrix->var_count;
	else
		threads = (matrix->thread_count >= max_threads)? max_threads : matrix->thread_count;
	*/
	/*if (blocks > prop.multiProcessorCount)
		error_exit(3, (char*)"CUDA Device out of shared memory!");*/

	StartTimer();
	{
		naive_aproach_one_thread<<<blocks, threads>>>(matrix);
		checkCudaErrors(cudaDeviceSynchronize());
	};

	//assert(threads*sizeof(double) <= prop.sharedMemPerBlock);
	double time = GetTimer()/1000;

	//check_solutions<<<1,threads, threads*sizeof(double)>>>(matrix->var_count, matrix->d_solution, matrix->d_reference);
	//checkCudaErrors(cudaDeviceSynchronize());
	return time;
} 

__global__ void naive_aproach_one_thread(cuda_matrix *matrix){
	int tid = threadIdx.x,
		block_id = gridDim.x;
	cuda_deque U, L;
	u_int w = matrix->window_size;
	double *a = matrix->d_matrix,
		   *maxval = matrix->d_maxval,
		   *minval = matrix->d_minval;

	if (tid == 0 && block_id == 0) //only one thread
	{
		for(u_int i=1; i < matrix->arrlen; ++i){
			if(i >= w){
				maxval[i-w] = a[U.size() > 0 ? U.front():i-1];
				minval[i-w] = a[L.size() > 0 ? L.front():i-1];
			}

			if(a[i] > a[i-1]){
				L.push_back(i-1);
				if(i == w + L.front())
					L.pop_front();

				while(U.size() > 0){
					if(a[i] <= a[U.back()]){
						if(i == w + U.front())
							U.pop_front();
						break;
					}
					U.pop_back();
				}
			}else{
				U.push_back(i-1);
				if(i == w + U.front())
					U.pop_front();
				
				while(L.size() > 0){
					if(a[i] >= a[L.back()]){
						if(i == w + L.front())
							L.pop_front();
						break;
					}
					L.pop_back();
				}
			}
		}
		maxval[matrix->arrlen - w] = a[U.size() > 0 ? U.front() : matrix->arrlen-1];
		minval[matrix->arrlen - w] = a[L.size() > 0 ? L.front() : matrix->arrlen-1];
	}
}