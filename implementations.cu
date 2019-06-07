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


double cuda_parallel_approach(cuda_matrix *matrix){
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks = matrix->core_count,
		threads = matrix->thread_count,
		max_threads = prop.maxThreadsPerBlock,
		max_sm = prop.multiProcessorCount;

	assert(max_threads >= threads);
	assert(max_sm >= blocks);

	//print_matrix(matrix->h_matrix, matrix->arrlen);
	checkCudaErrors(cudaDeviceSynchronize());
	StartTimer();
	{
		par_alg_inc_blocks<<<blocks, threads>>>(matrix->d_matrix, matrix->d_minval, matrix->d_maxval, matrix->arrlen, matrix->window_size);
		checkCudaErrors(cudaDeviceSynchronize());
		cudaError error = cudaMemcpy(matrix->h_maxval, matrix->d_maxval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
		error = cudaMemcpy(matrix->h_minval, matrix->d_minval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
	};

	double time = GetTimer()/1000;
	//print_matrix(matrix->h_minval, matrix->arrlen);
	//print_matrix(matrix->h_maxval, matrix->arrlen);
	printf("Time: %f\n", time);	
	return time;
}

#define DEC_FACTOR (10)
double thrust_approach(io_info *info) {

	int thrust_core_count = info->c_opt;
	int thrust_thread_count = info->t_opt;
	int thrust_window_size = info->w_opt;
	int thrust_arrlen = info->v_opt;
	unsigned int seed = info->seed;
	seed = (seed==0)? time(NULL) : seed;

	assert(thrust_window_size >= MIN_WIN_SIZE);

	thrust::host_vector<double> h_vec(thrust_arrlen); // generating a vector of wanted size
	//thrust::generate(h_vec.begin(), h_vec.end(), rand); // filling it with the random numbers, problem: generating same random values as before not possible
	//thrust::device_vector<int> d_vec = h_vec; // generating a vector on the gpu
	thrust::device_vector<double> thrust_minval(thrust_arrlen);
	thrust::device_vector<double> thrust_maxval(thrust_arrlen);

	//seed = (seed==0) ? time(NULL) : seed;
	for (int i = 0; i < thrust_arrlen; ++i){
		srand(seed*(i+1)*(thrust_arrlen+1)+i);
		h_vec[i] = ((double)rand()/(double)RAND_MAX)*DEC_FACTOR;
		assert(h_vec[i] > 0);
	}
	info->seed = seed;
	thrust::device_vector<double> d_vec = h_vec; // generating a vector on the gpu

	//inicijalizacija koristene graficke kartice
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks = thrust_core_count,
		threads = thrust_thread_count,
		max_threads = prop.maxThreadsPerBlock,
		max_sm = prop.multiProcessorCount;

	assert(max_threads >= threads);
	assert(max_sm >= blocks);
	//ovo sve moze ostati, nije nikakav problem

	
	/*for (int i=0; i<thrust_arrlen; i++) {
		std::cout << std::setprecision(1) << std::fixed << h_vec[i] << " ";
	}
	printf("\n");*/
	checkCudaErrors(cudaDeviceSynchronize());
	StartTimer();
	for (int i=0; i<(thrust_arrlen-thrust_window_size)+1; i++) {
		thrust::detail::normal_iterator< thrust::device_ptr<double> > minimum = thrust::min_element(thrust::device, d_vec.begin()+i, d_vec.begin()+i+thrust_window_size);
		thrust::detail::normal_iterator< thrust::device_ptr<double> > maximum = thrust::max_element(thrust::device, d_vec.begin()+i, d_vec.begin()+i+thrust_window_size);
		thrust_minval[i] = *minimum;
		thrust_maxval[i] = *maximum;
	}
	checkCudaErrors(cudaDeviceSynchronize());
	double time = GetTimer()/1000;

	/*for (int i=0; i<thrust_arrlen; i++) {
		std::cout << std::setprecision(1) << std::fixed << thrust_minval[i] << " ";
	}
	printf("\n");*/

	/*for (int i=0; i<thrust_arrlen; i++) {
		std::cout << std::setprecision(1) << std::fixed << thrust_maxval[i] << " ";
	}
	printf("\n");*/
	
	printf("Time: %f\n", time);	
	return time;
}

#define DEC_FACTOR (10)
double thrust_approach_amar(cuda_matrix *matrix, io_info *info) {
	/*
	int thrust_core_count = info->c_opt;
	int thrust_thread_count = info->t_opt;
	int thrust_window_size = info->w_opt;
	int thrust_arrlen = info->v_opt;
	unsigned int seed = info->seed;
	seed = (seed==0)? time(NULL) : seed;

	assert(thrust_window_size >= MIN_WIN_SIZE);

	thrust::host_vector<double> h_vec(thrust_arrlen); // generating a vector of wanted size
	//thrust::generate(h_vec.begin(), h_vec.end(), rand); // filling it with the random numbers, problem: generating same random values as before not possible
	//thrust::device_vector<int> d_vec = h_vec; // generating a vector on the gpu
	thrust::device_vector<double> thrust_minval(thrust_arrlen);
	thrust::device_vector<double> thrust_maxval(thrust_arrlen);

	//seed = (seed==0) ? time(NULL) : seed;
	for (int i = 0; i < thrust_arrlen; ++i){
		srand(seed*(i+1)*(thrust_arrlen+1)+i);
		h_vec[i] = ((double)rand()/(double)RAND_MAX)*DEC_FACTOR;
		assert(h_vec[i] > 0);
	}
	info->seed = seed;
	thrust::device_vector<double> d_vec = h_vec; // generating a vector on the gpu
	*/
	//inicijalizacija koristene graficke kartice
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks = matrix->core_count,
		threads = matrix->thread_count,
		max_threads = prop.maxThreadsPerBlock,
		max_sm = prop.multiProcessorCount;

	assert(max_threads >= threads);
	assert(max_sm >= blocks);
	//ovo sve moze ostati, nije nikakav problem


	thrust::device_ptr<double> matrix_ptr = thrust::device_pointer_cast(matrix->d_matrix);
	
	print_matrix(matrix->h_matrix, matrix->arrlen);
	checkCudaErrors(cudaDeviceSynchronize());
	StartTimer();
	{
		par_alg_thrust<<<blocks, threads>>>(matrix_ptr, matrix->d_minval, matrix->d_maxval, matrix->arrlen, matrix->window_size);
		checkCudaErrors(cudaDeviceSynchronize());
		cudaError error = cudaMemcpy(matrix->h_maxval, matrix->d_maxval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
		error = cudaMemcpy(matrix->h_minval, matrix->d_minval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaErrors(error);
	};
	double time = GetTimer()/1000;
	print_matrix(matrix->h_minval, matrix->arrlen);
	print_matrix(matrix->h_maxval, matrix->arrlen);

	printf("Time: %f\n", time);	
	return time;
}


double sequential_approach(cuda_matrix *matrix){
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, DEV_ID));

	int blocks,
		threads = matrix->thread_count,
		max_threads = prop.maxThreadsPerBlock,
		max_sm = prop.multiProcessorCount;

	blocks = matrix->core_count;

	assert(max_threads >= threads);
	assert(max_sm >= blocks);

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
	printf("Time: %f\n", time);
	return time;
} 

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
		min = max = matrix[addr_offs];
		for (int i = addr_offs + 1; i < addr_offs + window_size; ++i)
		{
			min = (matrix[i] < min)? matrix[i] : min;
			max = (matrix[i] > max)? matrix[i] : max;
		}
		assert(minval[addr_offs] == 0.0); //shows if there is overlapping
		minval[addr_offs] = min;
		maxval[addr_offs] = max;
		addr_offs += blockDim.x*gridDim.x;
	}
}

__global__ void par_alg_thrust(thrust::device_ptr<double> matrix_ptr, double *minval, double *maxval, int arrlen, int window_size){
	int tid = threadIdx.x,
		bid = blockIdx.x;
	assert(window_size >= MIN_WIN_SIZE);

	int addr_offs = tid + bid*blockDim.x;
	while(addr_offs+window_size < arrlen + 1) {
		thrust::detail::normal_iterator< thrust::device_ptr<double> > minimum = thrust::min_element(thrust::device, matrix_ptr+addr_offs, matrix_ptr+addr_offs+window_size);
		thrust::detail::normal_iterator< thrust::device_ptr<double> > maximum = thrust::max_element(thrust::device, matrix_ptr+addr_offs, matrix_ptr+addr_offs+window_size);
		
		double max = maximum[0],
			   min = minimum[0];
		assert(minval[addr_offs] == 0.0); //shows if there is overlapping
		minval[addr_offs] = min;
		maxval[addr_offs] = max;
		addr_offs += blockDim.x*gridDim.x;
	}
}
