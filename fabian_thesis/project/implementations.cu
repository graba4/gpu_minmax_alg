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

__device__ void divide_multi(double *arr, int length, int index);
__global__ void naive_aproach_cuda(double *d_matrix, size_t pitch, int var_count);
__device__ void print_matrix(double *d_matrix, int height, size_t pitch);
__global__ void calc_solutions(double *d_matrix, size_t pitch, int var_count,
				double *d_solution);
__global__ void calc_solutions_pipeline(Lock *locks, double *d_matrix, size_t pitch,
				int var_count, double *d_solution);
__global__ void check_solutions(int var_count, double *d_solution, double *d_reference);
__global__ void naive_aproach_divide_n_merch_cuda(double *d_matrix, size_t pitch, int var_count);
__global__ void naive_aproach_divide_n_merch_cuda_new(double *d_matrix, size_t pitch, int var_count,
				int start, int stop);
__global__ void naive_aproach_divide_n_merch_shared_cuda(double *d_matrix, size_t pitch,
				int var_count);

double naive_aproach(cuda_matrix *matrix){
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks,
		threads,
		max_threads = prop.maxThreadsPerBlock;

	blocks = matrix->core_count; //(matrix->var_count >= max_threads)? (matrix->var_count/max_threads)+1 : 1;
	if (matrix->thread_count < 0)
		threads = (matrix->var_count >= max_threads)? max_threads : matrix->var_count;
	else
		threads = (matrix->thread_count >= max_threads)? max_threads : matrix->thread_count;

	/*if (blocks > prop.multiProcessorCount)
		error_exit(3, (char*)"CUDA Device out of shared memory!");*/

	StartTimer();

	{
		int b_temp = blocks;
		while(true){
			naive_aproach_divide_n_merch_cuda<<<b_temp, threads>>>(matrix->d_matrix, matrix->pitch, matrix->var_count);
			//checkCudaErrors(cudaDeviceSynchronize());
			if (b_temp <= 1)
				break;
			b_temp = ROUND_UP(b_temp, 2);
		}
	};

	assert(threads*sizeof(double) <= prop.sharedMemPerBlock);
	
	calc_solutions<<<1, threads, threads*sizeof(double)>>>(matrix->d_matrix, matrix->pitch, matrix->var_count, matrix->d_solution);
	checkCudaErrors(cudaDeviceSynchronize());
	double time = GetTimer()/1000;

	check_solutions<<<1,threads, threads*sizeof(double)>>>(matrix->var_count, matrix->d_solution, matrix->d_reference);
	checkCudaErrors(cudaDeviceSynchronize());
	//delete locks;
	return time; 
}

double naive_aproach_pipe(cuda_matrix *matrix){
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks,
		threads,
		max_threads = prop.maxThreadsPerBlock;

	blocks = matrix->core_count; //(matrix->var_count >= max_threads)? (matrix->var_count/max_threads)+1 : 1;
	if (matrix->thread_count < 0)
		threads = (matrix->var_count >= max_threads)? max_threads : matrix->var_count;
	else
		threads = (matrix->thread_count >= max_threads)? max_threads : matrix->thread_count;

	Lock *locks_h = new Lock[matrix->var_count]; //(Lock*)malloc(sizeof(Lock)*matrix->var_count);
	Lock *locks;
	checkCudaErrors(cudaMalloc((void**)&locks, sizeof(Lock)*(matrix->var_count)));

	checkCudaErrors(cudaMemcpy(locks, locks_h, sizeof(Lock)*(matrix->var_count), cudaMemcpyHostToDevice));
	/*if (blocks > prop.multiProcessorCount)
		error_exit(3, (char*)"CUDA Device out of shared memory!");*/

	StartTimer();

	{
		int b_temp = blocks;
		while(true){
			naive_aproach_divide_n_merch_cuda<<<b_temp, threads>>>(matrix->d_matrix, matrix->pitch, matrix->var_count);
			//checkCudaErrors(cudaDeviceSynchronize());
			if (b_temp <= 1)
				break;
			b_temp = ROUND_UP(b_temp, 2);
		}
	};

	assert(threads*sizeof(double) <= prop.sharedMemPerBlock);
	
	calc_solutions_pipeline<<<blocks, threads, threads*sizeof(double)>>>(locks, matrix->d_matrix, matrix->pitch, matrix->var_count, matrix->d_solution);
	checkCudaErrors(cudaDeviceSynchronize());
	double time = GetTimer()/1000;

	check_solutions<<<1,threads, threads*sizeof(double)>>>(matrix->var_count, matrix->d_solution, matrix->d_reference);
	checkCudaErrors(cudaDeviceSynchronize());
	//delete locks;
	return time; 
}

double naive_aproach_decrement(cuda_matrix *matrix){
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks,
		threads,
		max_threads = prop.maxThreadsPerBlock;

	blocks = matrix->core_count; //(matrix->var_count >= max_threads)? (matrix->var_count/max_threads)+1 : 1;
	if (matrix->thread_count < 0)
		threads = (matrix->var_count >= max_threads)? max_threads : matrix->var_count;
	else
		threads = (matrix->thread_count >= max_threads)? max_threads : matrix->thread_count;


	/*if (blocks > prop.multiProcessorCount)
		error_exit(3, (char*)"CUDA Device out of shared memory!");*/

	StartTimer();

	while(blocks != 0) { //maybe better?
		naive_aproach_divide_n_merch_cuda<<<blocks, threads>>>(matrix->d_matrix, matrix->pitch, matrix->var_count);
		blocks--;
	}

	assert(threads*sizeof(double) <= prop.sharedMemPerBlock);
	calc_solutions<<<1, threads, threads*sizeof(double)>>>(matrix->d_matrix, matrix->pitch, matrix->var_count, matrix->d_solution);
	checkCudaErrors(cudaDeviceSynchronize());

	double time = GetTimer()/1000;

	check_solutions<<<1,threads, threads*sizeof(double)>>>(matrix->var_count, matrix->d_solution, matrix->d_reference);
	checkCudaErrors(cudaDeviceSynchronize());
	return time;
}

double naive_aproach_decrement_new(cuda_matrix *matrix){
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks,
		threads,
		max_threads = prop.maxThreadsPerBlock;

	blocks = matrix->core_count; //(matrix->var_count >= max_threads)? (matrix->var_count/max_threads)+1 : 1;
	if (matrix->thread_count < 0)
		threads = (matrix->var_count >= max_threads)? max_threads : matrix->var_count;
	else
		threads = (matrix->thread_count >= max_threads)? max_threads : matrix->thread_count;


	/*if (blocks > prop.multiProcessorCount)
		error_exit(3, (char*)"CUDA Device out of shared memory!");*/

	StartTimer();

	for (int start = 0, offset = matrix->var_count; start < matrix->var_count-1; start+=offset)
	{
		blocks = matrix->core_count;
		offset = ROUND_UP(offset, 2);
		while(blocks != 0) { //maybe better?
			naive_aproach_divide_n_merch_cuda_new<<<blocks, threads>>>(matrix->d_matrix, matrix->pitch,
				matrix->var_count, start, start+offset);
			blocks--;
		}
	}
	

	assert(threads*sizeof(double) <= prop.sharedMemPerBlock);
	calc_solutions<<<1, threads, threads*sizeof(double)>>>(matrix->d_matrix, matrix->pitch, matrix->var_count, matrix->d_solution);
	checkCudaErrors(cudaDeviceSynchronize());

	double time = GetTimer()/1000;

	check_solutions<<<1,threads, threads*sizeof(double)>>>(matrix->var_count, matrix->d_solution, matrix->d_reference);
	checkCudaErrors(cudaDeviceSynchronize());
	return time;
}

double naive_aproach_decrement_pipe(cuda_matrix *matrix){
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks,
		threads,
		max_threads = prop.maxThreadsPerBlock;

	blocks = matrix->core_count; //(matrix->var_count >= max_threads)? (matrix->var_count/max_threads)+1 : 1;
	if (matrix->thread_count < 0)
		threads = (matrix->var_count >= max_threads)? max_threads : matrix->var_count;
	else
		threads = (matrix->thread_count >= max_threads)? max_threads : matrix->thread_count;

	Lock *locks_h = new Lock[matrix->var_count]; //(Lock*)malloc(sizeof(Lock)*matrix->var_count);
	Lock *locks;
	checkCudaErrors(cudaMalloc((void**)&locks, sizeof(Lock)*(matrix->var_count)));

	checkCudaErrors(cudaMemcpy(locks, locks_h, sizeof(Lock)*(matrix->var_count), cudaMemcpyHostToDevice));
	/*if (blocks > prop.multiProcessorCount)
		error_exit(3, (char*)"CUDA Device out of shared memory!");*/

	StartTimer();

	{
		int b_temp = blocks;
		while(b_temp != 0) { //maybe better?
				naive_aproach_divide_n_merch_cuda<<<b_temp, threads>>>(matrix->d_matrix, matrix->pitch, matrix->var_count);
				b_temp--;
		}
	};

	assert(threads*sizeof(double) <= prop.sharedMemPerBlock);
	
	calc_solutions_pipeline<<<blocks, threads, threads*sizeof(double)>>>(locks, matrix->d_matrix, matrix->pitch, matrix->var_count, matrix->d_solution);
	checkCudaErrors(cudaDeviceSynchronize());
	double time = GetTimer()/1000;

	check_solutions<<<1,threads, threads*sizeof(double)>>>(matrix->var_count, matrix->d_solution, matrix->d_reference);
	checkCudaErrors(cudaDeviceSynchronize());
	//delete locks;
	return time; 
}

__device__ void divide_multi(double *arr, int length, int index){
	assert((index < length));
	assert((index >= 0));
	int tid = threadIdx.x + index;

	double div = arr[index];
	__syncthreads();
	while (tid < length){
		arr[tid] /= div;
		tid += blockDim.x;
	}
}

__global__ void naive_aproach_divide_n_merch_cuda(double *d_matrix, size_t pitch, int var_count){
	int tid = threadIdx.x;
	int block_inc = gridDim.x;

	int count = 0;
	for (int i = blockIdx.x; i < var_count; i += block_inc)
	{
		double *row = (double*)((char*)d_matrix + i * pitch);
		divide_multi(row, var_count+1, count);
		for (int y = i; y < var_count-block_inc; y += block_inc)
		{
			double *row_y = (double*)((char*)d_matrix + (y+block_inc) * pitch);
			double factor = row_y[count];
		    __syncthreads();
			if (factor == 0) //given matrix does not contain any 0
				break;

		    int tid_temp = tid;
			while (tid_temp < var_count+1)
			{
				row_y[tid_temp] -= row[tid_temp] * factor;
				tid_temp += blockDim.x;
			}
		}
		
		count++;
	}
}

__global__ void naive_aproach_divide_n_merch_cuda_new(double *d_matrix, size_t pitch, int var_count, int start, int stop){
	int tid = threadIdx.x;
	int block_inc = gridDim.x;

	int count = start;
	for (int i = blockIdx.x+start; i < var_count; i += block_inc)
	{
		if (block_inc == 1 && i >= stop)
			return;

		double *row = (double*)((char*)d_matrix + i * pitch);
		divide_multi(row, var_count+1, count);
		for (int y = i; y < var_count-block_inc; y += block_inc)
		{
			double *row_y = (double*)((char*)d_matrix + (y+block_inc) * pitch);
			double factor = row_y[count];

			__syncthreads();
			if (factor == 0) //given matrix does not contain any 0
				break;
			
		    int tid_temp = tid;
			while (tid_temp < var_count+1)
			{
				row_y[tid_temp] -= row[tid_temp] * factor;
				tid_temp += blockDim.x;
			}
		}
		
		count++;
	}
}

__device__ void print_matrix(double *d_matrix, int height, size_t pitch){
	assert(d_matrix != NULL);
	int tid = threadIdx.x + blockIdx.x;
	if(tid == 0){
		for (int j = 0; j < height; j++) {
			// image row
			double *row = (double*)((char*)d_matrix + j * pitch);
			for (int i = 0; i < height+1; i++){
				if (i == height)
					printf("|%.1f", (row[i] == -0.0)? 0.0 : row[i]);
				else
					printf("%.1f ", (row[i] == -0.0)? 0.0 : row[i]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

__global__ void calc_solutions(double *d_matrix, size_t pitch, int var_count, double *d_solution){
	int tid = blockDim.x*blockIdx.x + threadIdx.x,
		count = 0;
	for (int y = var_count-1; y >= 0; --y)
	{
		int tid_temp = tid + (var_count - blockDim.x);
		double *temp_p = (double*)((char*)d_matrix + y * pitch);
		double temp = 0;
		extern __shared__ double cache[];
		__syncthreads();

		while(BETWEEN(tid_temp, (var_count-1) - count, var_count)){
			assert(tid_temp < var_count+1);
			assert(tid_temp >= 0);
			temp += temp_p[tid_temp]*d_solution[tid_temp];
			tid_temp -= blockDim.x;
		}

		assert(threadIdx.x < blockDim.x);
		cache[threadIdx.x] = temp;
		__syncthreads();

		int i = ROUND_UP(blockDim.x, 2); //cite cuda by example page 85
		while(i > 1){
			if (threadIdx.x < i)
			{
				if (threadIdx.x +i < blockDim.x)
				{
					cache[threadIdx.x] += cache[threadIdx.x +i];
					cache[threadIdx.x +i] = 0;
				}
			}
			__syncthreads();

			i = ROUND_UP(i, 2);
		}

		if(threadIdx.x == 0){
			if (i == 1)
				cache[0] += cache[1];
			assert(var_count-count-1 < var_count);
			assert(var_count-count-1 >= 0);
			d_solution[var_count-count-1] = temp_p[var_count] - cache[0];
		}

		__syncthreads();

		count++; 
	}
}

__global__ void calc_solutions_pipeline(Lock *locks, double *d_matrix, size_t pitch, int var_count, double *d_solution){
	int tid = threadIdx.x,
		count,
		block_size = var_count/gridDim.x,
		start_pos;

	start_pos = (var_count-1) - block_size*blockIdx.x;
	block_size += (blockIdx.x == (gridDim.x-1))? var_count%gridDim.x : 0;
	count = var_count-(start_pos+1);
	if (blockIdx.x == 0)
	{
		for (int y = start_pos; y > start_pos-block_size; --y)
		{
			int tid_temp = tid + (var_count - blockDim.x);
			double *temp_p = (double*)((char*)d_matrix + y * pitch);
			double temp = 0;
			extern __shared__ double cache[];
			__syncthreads();
			//printf("1tid: %d\n", tid_temp);

			while(BETWEEN(tid_temp, (var_count-1) - count, var_count)){
				assert(tid_temp < var_count+1);
				assert(tid_temp >= 0);
				temp += temp_p[tid_temp]*d_solution[tid_temp];
				tid_temp -= blockDim.x;
			}

			assert(threadIdx.x < blockDim.x);
			cache[threadIdx.x] = temp;
			__syncthreads();

			int i = ROUND_UP(blockDim.x, 2); //cite cuda by example page 85
			while(i > 1){
				if (threadIdx.x < i)
				{
					if (threadIdx.x +i < blockDim.x)
					{
						cache[threadIdx.x] += cache[threadIdx.x +i];
						cache[threadIdx.x +i] = 0;
					}
				} 
				__syncthreads();
				
				i = ROUND_UP(i, 2);
			}

			if(threadIdx.x == 0){
				if (i == 1)
					cache[0] += cache[1];
				assert(var_count-count-1 < var_count);
				assert(var_count-count-1 >= 0);
				d_solution[var_count-count-1] = temp_p[var_count] - cache[0];
				//printf("release %d\n", y);
				locks[y].unlock();
			}

			__syncthreads();
			
			count++;
		}
	} else {
		for (int y = start_pos; y > start_pos-block_size; --y)
		{
			int tid_temp = tid + (var_count - blockDim.x);
			double *temp_p = (double*)((char*)d_matrix + y * pitch);
			double temp = 0;
			extern __shared__ double cache[];

			__syncthreads();
 
			while(BETWEEN(tid_temp, (var_count-1) - count, var_count)){
				assert(tid_temp < var_count+1);
				assert(tid_temp >= 0); 
				while (locks[tid_temp].is_free());
				
				temp += temp_p[tid_temp]*d_solution[tid_temp];
				tid_temp -= blockDim.x;
			}

			assert(threadIdx.x < blockDim.x);
			cache[threadIdx.x] = temp;
			__syncthreads();

			int i = ROUND_UP(blockDim.x, 2); //cite cuda by example page 85
			while(i > 1){
				if (threadIdx.x < i)
				{
					if (threadIdx.x +i < blockDim.x)
					{
						cache[threadIdx.x] += cache[threadIdx.x +i];
						cache[threadIdx.x +i] = 0;
					}
				}
				__syncthreads();

				i = ROUND_UP(i, 2);
			}

			if(threadIdx.x == 0){
				if (i == 1)
					cache[0] += cache[1];
				assert(var_count-count-1 < var_count);
				assert(var_count-count-1 >= 0);
				d_solution[var_count-count-1] = temp_p[var_count] - cache[0];
				/*printf("%d\n", y);
				printf("release %d blockidx: %d\n", y, blockIdx.x);*/
				locks[y].unlock();
			}

			__syncthreads();


			count++;
		}
	}
}

#define DEVIATION (0.001) //in %
__global__ void check_solutions(int var_count, double *d_solution, double *d_reference){
	int tid = threadIdx.x, tid_temp;
	double dev;
	extern __shared__ double cache[];

	tid_temp = tid;
	cache[tid] = 0;
	while(tid_temp < var_count)
	{
		cache[tid] += d_reference[tid_temp] * d_solution[tid_temp];
		tid_temp+=blockDim.x;
	}
	__syncthreads();

	int i = ROUND_UP(blockDim.x, 2); //cite cuda by example page 85
	while(i > 1){
		if (tid < i)
		{
			if (tid +i < blockDim.x)
			{
				cache[tid] += cache[tid +i];
				cache[tid+i] = 0;
			}
		}
		__syncthreads();

		i = ROUND_UP(i, 2);
	}

	if(tid == 0){
		double sum = (i == 1)? cache[0]+cache[1] : cache[0];

		//printf("sum: %.25f d_reference: %.25f\n", sum, d_reference[var_count]);
		dev = ((d_reference[var_count] - sum)*100)*sum;
		dev = (dev < 0) ? dev*(-1) : dev;
		//printf("%.30f\n", dev);
		assert(dev < DEVIATION);
	}
}