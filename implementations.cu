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

__global__ void naive_aproach_one_thread(double *matrix, double *minval, double *maxval, int arrlen, int window_size);

__global__ void naive_aproach_multi_thread(double *matrix, double *minval, double *maxval, int arrlen, int window_size);

class Min_Max_calc
{
private:
	double min,
		   max;

public:
	__device__ Min_Max_calc(double *arr, unsigned int length){
		assert(arr != NULL);
		min = max = arr[0];
		for (int i = 1; i < length; ++i)
		{
			min = (arr[i] < min)? arr[i] : min;
			max = (arr[i] > max)? arr[i] : max;
		}
	}
	__device__ ~Min_Max_calc();

	__device__ double getMin(){
		return min;
	}

	__device__ double getMax(){
		return max;
	}
};

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
		naive_aproach_one_thread<<<blocks, threads>>>(matrix->d_matrix, matrix->d_minval, matrix->d_maxval, matrix->arrlen, matrix->window_size);
		checkCudaErrors(cudaDeviceSynchronize());
	};

	//assert(threads*sizeof(double) <= prop.sharedMemPerBlock);
	double time = GetTimer()/1000;

	//check_solutions<<<1,threads, threads*sizeof(double)>>>(matrix->var_count, matrix->d_solution, matrix->d_reference);
	//checkCudaErrors(cudaDeviceSynchronize());
	printf("Time: %f\n", time);
	return time;
}

double naive_aproach_amar(cuda_matrix *matrix){
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
		naive_aproach_multi_thread<<<blocks, threads>>>(matrix->d_matrix, matrix->d_minval, matrix->d_maxval, matrix->arrlen, matrix->window_size);
		checkCudaErrors(cudaDeviceSynchronize());
	};

	//assert(threads*sizeof(double) <= prop.sharedMemPerBlock);
	double time = GetTimer()/1000;

	//check_solutions<<<1,threads, threads*sizeof(double)>>>(matrix->var_count, matrix->d_solution, matrix->d_reference);
	//checkCudaErrors(cudaDeviceSynchronize());
	printf("Time: %f\n", time);
	return time;
} 

__device__ void print_matrixx(double *matrix, int length)
{
	assert(matrix != NULL);

	/* image row */
	for (int i = 0; i < length; i++){
		printf("%.1f ", (matrix[i] == -0.0)? 0.0 : matrix[i]);
	}
	printf("\n");
}

#define MIN_WIN_SIZE (3)
__global__ void naive_aproach_one_thread(double *matrix, double *minval, double *maxval, int arrlen, int window_size){
	int tid = threadIdx.x,
		block_id = gridDim.x;
	assert(window_size >= MIN_WIN_SIZE);
	cuda_deque U, L;
	double *a = matrix;
	//print_matrixx(a, arrlen);

	if (tid == 0 && block_id == 1) //only one thread
	{
		for(u_int i = 1; i < arrlen; ++i){
			if(i >= window_size){
				maxval[i-window_size] = a[U.size() > 0 ? U.front():i-1];
				minval[i-window_size] = a[L.size() > 0 ? L.front():i-1];
			}

			if(a[i] > a[i-1]){
				L.push_back(i-1);
				
				if(i == window_size + L.front())
					L.pop_front();

				while(U.size() > 0){
					if(a[i] <= a[U.back()]){
						if(i == window_size + U.front())
							U.pop_front();
						break;
					}
					U.pop_back();
				}
			}else{
				U.push_back(i-1);
				
				if(i == window_size + U.front())
					U.pop_front();
				
				
				while(L.size() > 0){
					if(a[i] >= a[L.back()]){
						if(i == window_size + L.front())
							L.pop_front();
						break;
					}
					L.pop_back();
				}
			}
		}
		maxval[arrlen - window_size] = a[U.size() > 0 ? U.front() : arrlen-1];
		minval[arrlen - window_size] = a[L.size() > 0 ? L.front() : arrlen-1];
		
		//print_matrixx(maxval, arrlen);	
		//print_matrixx(minval, arrlen);
		
	}
}

__global__ void par_alg_inc_win(double *matrix, double *minval, double *maxval, int arrlen, int window_size){
	int tid = threadIdx.x,
		bid = blockIdx.x,
		shift_amount = 0;
	assert(window_size >= MIN_WIN_SIZE);

	int addr_offs = (tid + shift_amount) + bid*window_size*2;
	while(addr_offs+window_size < arrlen) {
		Min_Max_calc m((double *)(matrix + addr_offs));
		minval[addr_offs] = m.getMin();
		maxval[addr_offs] = m.getMax();

		addr_offs += gridDim.x;
	}
}

__global__ void par_alg_inc_blocks(double *matrix, double *minval, double *maxval, int arrlen, int window_size){
	int tid = threadIdx.x,
		bid = blockIdx.x,
		shift_amount = 0;
	assert(window_size >= MIN_WIN_SIZE);

	int addr_offs = (tid + shift_amount) + bid*window_size;
	while(addr_offs+window_size < arrlen) {
		Min_Max_calc m((double *)(matrix + addr_offs));
		minval[addr_offs] = m.getMin();
		maxval[addr_offs] = m.getMax();

		addr_offs += window_size;
	}
}

__global__ void naive_aproach_multi_thread(double *matrix, double *minval, double *maxval, int arrlen, int window_size){
	int tid = threadIdx.x,
		block_id = gridDim.x;
	cuda_deque U, L;
	double *a = matrix;
	//print_matrixx(a, arrlen);
	//printf("%d %d\n", tid, block_id);
	if(tid == 0){
		for(u_int i = 1; i < arrlen; ++i){
			if(i >= window_size){
				maxval[i-window_size] = a[U.size() > 0 ? U.front():i-1];
			}				

			if(a[i] > a[i-1]){
				while(U.size() > 0){
					if(a[i] <= a[U.back()]){
						if(i == window_size + U.front())
							U.pop_front();
						break;
					}
					U.pop_back();
				}
			}else{
				U.push_back(i-1);				
				if(i == window_size + U.front())
					U.pop_front();
			}
		}
		maxval[arrlen - window_size] = a[U.size() > 0 ? U.front() : arrlen-1];
		//printf("%d %d\n", tid, block_id);	
	}
	if(tid == 1){
		//printf("%d %d\n", tid, block_id);	
		for(u_int i = 1; i < arrlen; ++i){
			if(i >= window_size){
				minval[i-window_size] = a[L.size() > 0 ? L.front():i-1];
			}
			if(a[i] > a[i-1]){
				L.push_back(i-1);
				
				if(i == window_size + L.front())
					L.pop_front();

			}else{
				while(L.size() > 0){
					if(a[i] >= a[L.back()]){
						if(i == window_size + L.front())
							L.pop_front();
						break;
					}
					L.pop_back();
				}
			}
		}
		minval[arrlen - window_size] = a[L.size() > 0 ? L.front() : arrlen-1];		
		//printf("%d %d\n", tid, block_id);
	}
	//print_matrixx(maxval, arrlen);	
	//print_matrixx(minval, arrlen);
}