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
#include <iomanip>
#include <ctime>

#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


#define ROUND_UP(N, S) (N%S == 0) ? N/S : N/S+1
#define BETWEEN(value, min, max) (value < max && value > min)
#define DEV_ID (0)
#define MIN_WIN_SIZE (3)

__global__ void lemire_one_thread(double *matrix, double *minval, double *maxval, int arrlen, int window_size);

__global__ void par_alg_inc_blocks(double *matrix, double *minval, double *maxval, int arrlen, int window_size);

__global__ void par_alg_thrust(thrust::device_ptr<double> matrix, double *minval, double *maxval, int arrlen, int window_size);

__device__ void print_matrixx(double *matrix, int length);

struct arbitrary_functor
{
	int window_size;
    template <typename Tuple>
    __host__ __device__
    void operator()(const Tuple& t)
    {
        double* d_first = &thrust::get<0>(t);

        double *min = thrust::min_element(thrust::device, d_first, d_first + window_size);
    	thrust::get<1>(t) = min[0];
    	double *max = thrust::max_element(thrust::device, d_first, d_first + window_size);
    	thrust::get<2>(t) = max[0];
    }
};

__device__ void print_matrixx(double *matrix, int length)
{
	assert(matrix != NULL);
	__syncthreads();
	if(blockIdx.x == 0 && threadIdx.x == 0){
		/* image row */
		for (int i = 0; i < length; i++){
			printf("%.1f ", (matrix[i] == -0.0)? 0.0 : matrix[i]);
		}
		printf("\n");
	}
}

double cuda_parallel_approach(cuda_matrix *matrix){
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks = matrix->core_count,
		threads = matrix->thread_count,
		max_threads = prop.maxThreadsPerBlock,
		max_sm = prop.multiProcessorCount;

	assert(max_threads >= threads);
	assert(max_sm >= blocks);

	checkCudaErrors(cudaDeviceSynchronize());
	StartTimer();
	clock_t begin = clock();
	{
		par_alg_inc_blocks<<<blocks, threads>>>(matrix->d_matrix, matrix->d_minval, matrix->d_maxval, matrix->arrlen, matrix->window_size);
		checkCudaErrors(cudaDeviceSynchronize());
		cudaError error = cudaMemcpy(matrix->h_maxval, matrix->d_maxval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
		error = cudaMemcpy(matrix->h_minval, matrix->d_minval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
	};

	double time = GetTimer()/1000;
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Cuda synchronize alg time: %f\n", time_spent);	

	return time_spent;
}

#define DEC_FACTOR (10)
double thrust_approach(cuda_matrix *matrix) {
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int thrust_core_count = matrix->core_count;
	int thrust_thread_count = matrix->thread_count;
	int thrust_window_size = matrix->window_size;
	int thrust_arrlen = matrix->arrlen;

	int blocks = thrust_core_count,
		threads = thrust_thread_count,
		max_threads = prop.maxThreadsPerBlock,
		max_sm = prop.multiProcessorCount;

	assert(max_threads >= threads);
	assert(max_sm >= blocks);

	assert(thrust_window_size >= MIN_WIN_SIZE);

	checkCudaErrors(cudaDeviceSynchronize());
	StartTimer();
	clock_t begin = clock();
	{
	  	arbitrary_functor arb;
	  	arb.window_size = thrust_window_size;

	  	thrust::device_vector<double> thrust_minval(thrust_arrlen);
		thrust::device_vector<double> thrust_maxval(thrust_arrlen);

	  	thrust::device_ptr<double> matrix_ptr = thrust::device_pointer_cast(matrix->d_matrix);
	  	
	  	thrust::device_ptr<double> d_first = thrust::device_pointer_cast(matrix->d_matrix);
	  	thrust::device_ptr<double> d_last = thrust::device_pointer_cast(matrix->d_matrix) + thrust_arrlen - thrust_window_size + 1;
	  	thrust::device_ptr<double> min_first = thrust_minval.data();
	  	thrust::device_ptr<double> min_last = thrust_minval.data() + thrust_arrlen - thrust_window_size + 1;
	  	thrust::device_ptr<double> max_first = thrust_maxval.data();
	  	thrust::device_ptr<double> max_last = thrust_maxval.data() + thrust_arrlen - thrust_window_size + 1;

		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_first,
																 	  min_first,
																 	  max_first)),
						 thrust::make_zip_iterator(thrust::make_tuple(d_last,
						 										 	  min_last,
						 										 	  max_last)),
						 arb);



		checkCudaErrors(cudaDeviceSynchronize());
		thrust::copy(thrust_minval.begin(), thrust_minval.end(), thrust::device_pointer_cast(matrix->d_minval));
		thrust::copy(thrust_maxval.begin(), thrust_maxval.end(), thrust::device_pointer_cast(matrix->d_maxval));

		cudaError error = cudaMemcpy(matrix->h_maxval, matrix->d_maxval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
		error = cudaMemcpy(matrix->h_minval, matrix->d_minval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
	};
	double time = GetTimer()/1000;
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Thurst alg time: %f\n", time_spent);	
	return time_spent;
}

double streams_approach(io_info *info) {
	cudaError error;
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks, threads;
	int max_threads = prop.maxThreadsPerBlock,
		max_sm = prop.multiProcessorCount;

	int nStreams = 4;
	cudaStream_t streams[nStreams];

	info->v_opt = (int)(info->v_opt / 4);
	
	cuda_matrix *matrix[4];

	StartTimer();
	for(int i=0; i < nStreams; ++i) {
		cudaStreamCreate(&streams[i]);
		matrix[i] = allocate_recources_streams(info); //this needs to be changed, in create_matrix we should do cudaMemcpyAsync instead of cudaMemcpy
		blocks = matrix[i]->core_count;
		threads = matrix[i]->thread_count;
	}
	for(int i=0; i < nStreams; ++i) {
		par_alg_inc_blocks<<<blocks, threads, 0, streams[i]>>>(matrix[i]->d_matrix, matrix[i]->d_minval, matrix[i]->d_maxval, matrix[i]->arrlen, matrix[i]->window_size);
	}
	for(int i=0; i < nStreams; ++i) {
		error = cudaMemcpyAsync(matrix[i]->h_minval, matrix[i]->d_minval, matrix[i]->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
	}
	double time = GetTimer()/1000;

	for (int i=0; i<nStreams; ++i) {
		free_matrix(matrix[i]);
	}


	printf("Time: %f\n", time);	
	return time;
}

double sequential_approach(cuda_matrix *matrix){
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks,
		threads = matrix->thread_count;
		//max_threads = prop.maxThreadsPerBlock,
		//max_sm = prop.multiProcessorCount;

	blocks = matrix->core_count;

	//assert(max_threads >= threads);
	//assert(max_sm >= blocks);

	//print_matrix(matrix->h_matrix, matrix->arrlen);
	blocks = 1;
	threads = 1;
	checkCudaErrors(cudaDeviceSynchronize());
	StartTimer();
	{
		lemire_one_thread<<<blocks, threads>>>(matrix->d_matrix, matrix->d_minval, matrix->d_maxval, matrix->arrlen, matrix->window_size);
		checkCudaErrors(cudaDeviceSynchronize());
		cudaError error = cudaMemcpy(matrix->h_maxval, matrix->d_maxval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
		error = cudaMemcpy(matrix->h_minval, matrix->d_minval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
	};
	double time = GetTimer()/1000;
	//print_matrix(matrix->h_minval, matrix->arrlen);
	//print_matrix(matrix->h_maxval, matrix->arrlen);
	printf("Lemire alg time: %f\n", time);
	return time;
} 

#define MIN_WIN_SIZE (3)
__global__ void lemire_one_thread(double *matrix, double *minval, double *maxval, int arrlen, int window_size){
	assert(window_size >= MIN_WIN_SIZE);
	cuda_deque U, L;
	double *a = matrix;
	if (threadIdx.x == 0 && blockIdx.x == 0) //only one thread
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
	}
}

__global__ void par_alg_inc_blocks(double *matrix, double *minval, double *maxval, int arrlen, int window_size){
	int tid = threadIdx.x,
		bid = blockIdx.x;
	assert(window_size >= MIN_WIN_SIZE);

	int addr_offs = tid + bid*blockDim.x;
	while(addr_offs+window_size < arrlen + 1) {
		double min, max;
		assert(addr_offs < arrlen);
		min = max = matrix[addr_offs];
		for (int i = addr_offs + 1; i < addr_offs + window_size; ++i)
		{
			assert(i < arrlen);
			min = (matrix[i] < min)? matrix[i] : min;
			max = (matrix[i] > max)? matrix[i] : max;
		}
		assert(minval[addr_offs] == 0.0); //shows if there is overlapping
		minval[addr_offs] = min;
		maxval[addr_offs] = max;
		addr_offs += blockDim.x*gridDim.x;
	}
}
