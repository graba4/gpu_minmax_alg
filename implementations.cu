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
#define BETWEEN(value, min, max) (value <= max && value >= min)
#define DEV_ID (0)
#define MIN_WIN_SIZE (3)

__global__ void lemire_one_thread(double *matrix, double *minval, double *maxval, int arrlen, int window_size);

__global__ void par_alg_inc_blocks(double *matrix, double *minval, double *maxval, int arrlen, int window_size);

__global__ void par_alg_thrust(thrust::device_ptr<double> matrix, double *minval, double *maxval, int arrlen, int window_size);

__device__ void print_matrixx(double *matrix, int length);
__global__ void cuda_print_arr(double *arr, size_t len);

struct arbitrary_functor
{
	//using namespace thrust;
	int window_size;
    template <typename Tuple>
    __host__ __device__
    void operator()(const Tuple &t)
    {
    	using namespace thrust;
        double* d_first = &get<0>(t);

        double *min = min_element(device, d_first, d_first + window_size);
    	get<1>(t) = min[0];
    	double *max = max_element(device, d_first, d_first + window_size);
    	get<2>(t) = max[0];
    }
};

__device__ void print_matrixx(double *matrix, int length)
{
	assert(matrix != NULL);
	__syncthreads();
	if(blockIdx.x == 0 && threadIdx.x == 0){
		/* image row */
		for (int i = 0; i < length; i++){
			printf("%.0f ", (matrix[i] == -0.0)? 0.0 : matrix[i]);
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
	{
		par_alg_inc_blocks<<<blocks, threads>>>(matrix->d_matrix, matrix->d_minval, matrix->d_maxval, matrix->arrlen, matrix->window_size);
	};

	
	double time_spent = GetTimer()/1000;

	return time_spent;
}

#define DEC_FACTOR (10)
double thrust_approach(cuda_matrix *matrix) {
	using namespace thrust; //we dont have to write thrust:: anymore
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	assert(matrix->window_size >= MIN_WIN_SIZE);

	checkCudaErrors(cudaDeviceSynchronize());
	StartTimer();
	{
	  	arbitrary_functor arb;
	  	arb.window_size = matrix->window_size;
	  	
	  	device_ptr<double> d_first = device_pointer_cast(matrix->d_matrix);
	  	device_ptr<double> d_last = device_pointer_cast(matrix->d_matrix) + matrix->arrlen - matrix->window_size + 1;
	  	device_ptr<double> min_first = device_pointer_cast(matrix->d_minval);
	  	device_ptr<double> min_last = device_pointer_cast(matrix->d_minval) + matrix->arrlen - matrix->window_size + 1;
	  	device_ptr<double> max_first = device_pointer_cast(matrix->d_maxval);
	  	device_ptr<double> max_last = device_pointer_cast(matrix->d_maxval) + matrix->arrlen - matrix->window_size + 1;

		for_each(make_zip_iterator(make_tuple(d_first, min_first, max_first)),
				 make_zip_iterator(make_tuple(d_last, min_last, max_last)),
				 arb
		);
		checkCudaErrors(cudaDeviceSynchronize());
	};
	double time_spent = GetTimer()/1000;


	return time_spent;
}

double streams_approach(cuda_matrix *matrix) {
	cudaError error;
	//cudaDeviceProp prop;
	//checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	StartTimer();

	double time = GetTimer()/1000;

	cudaStream_t stream0, stream1;
	error = cudaStreamCreate(&stream0);
	checkCudaErrors(error);

	print_matrix(matrix->h_matrix, matrix->arrlen);
	for (int i = 0; i < matrix->arrlen; i+=CHUNK_SIZE)
	{
		size_t data_size = (matrix->arrlen-i < CHUNK_SIZE) ? matrix->arrlen-i : CHUNK_SIZE;

		error = cudaMemcpyAsync(matrix->d_matrix+i, (matrix->h_matrix)+i, data_size*sizeof(double), cudaMemcpyHostToDevice, stream0);
		checkCudaErrors(error);
		par_alg_inc_blocks<<<matrix->core_count, matrix->thread_count, 0, stream0>>>(matrix->d_matrix, matrix->d_minval, matrix->d_maxval, data_size, matrix->window_size);

		cuda_print_arr<<<1, 1, 0, stream0>>>(matrix->d_minval, matrix->arrlen);
	}
	cudaStreamSynchronize(stream0);

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
		//assert(minval[addr_offs] == 0.0); //shows if there is overlapping
		//assert(maxval[addr_offs] == 0.0); //shows if there is overlapping

		minval[addr_offs] = min;
		maxval[addr_offs] = max;
		addr_offs += blockDim.x*gridDim.x;
	}

	//print_matrixx(minval, 15);
}

__global__ void cuda_print_arr(double *arr, size_t len){
	print_matrixx(arr, len);
}