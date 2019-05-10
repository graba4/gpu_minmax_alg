#include <string.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "resources.h"
#include <stdlib.h>

__global__ void print_ref(double *ref, int len);
cuda_matrix* allocate_recources(io_info *info);
void free_matrix(double *matrix);
void free_matrix(cuda_matrix *matrix);
void create_matrix(cuda_matrix *matrix, int arrlen, bool clear, unsigned int seed);
void print_matrix(double *matrix, int length);
void gen_reference(cuda_matrix *matrix, double *h_matrix, int length);
void print_dev_info();

cuda_matrix* allocate_recources(io_info *info)
{
	cudaError error;
	int arrlen = info->v_opt;

	cuda_matrix *matrix = (cuda_matrix*)malloc(sizeof(cuda_matrix));
	assert(matrix != NULL);
	matrix->arrlen = arrlen;
	matrix->core_count = info->c_opt;
	matrix->thread_count = info->t_opt;
	matrix->window_size = info->w_opt;

	create_matrix(matrix, arrlen, false, info->seed);
	info->seed = matrix->seed;

	error = cudaMalloc(&(matrix->d_solution), sizeof(double)*arrlen);
	checkCudaErrors(error);

	return matrix;
}

#define DEC_FACTOR (1000)
void create_matrix(cuda_matrix *matrix, int arrlen, bool clear, unsigned int seed)
{
	cudaError error;
	double *h_matrix;

	if (clear) {
		h_matrix = (double *)calloc(arrlen, sizeof(double));
		assert(h_matrix != NULL);
	} else {
		h_matrix = (double *)malloc(sizeof(double) * arrlen);
		assert(h_matrix != NULL);
		seed = (seed==0)? time(NULL) : seed;
		matrix->seed = seed;

		for (int i = 0; i < arrlen; ++i){
			srand(seed*(i+1)*(arrlen+1)+i);
			h_matrix[i] = ((double)rand()/(double)RAND_MAX)*DEC_FACTOR;
			assert(h_matrix[i] > 0);
		}
	}

	//print_matrix(h_matrix, arrlen);

	error = cudaMalloc(&(matrix->d_matrix), sizeof(double)*arrlen);
	checkCudaErrors(error);

	error = cudaMalloc(&(matrix->d_minval), sizeof(double)*arrlen);
	checkCudaErrors(error);

	error = cudaMalloc(&(matrix->d_maxval), sizeof(double)*arrlen);
	checkCudaErrors(error);

	error = cudaMemcpy(matrix->d_matrix, h_matrix, arrlen*sizeof(double), cudaMemcpyHostToDevice);
	checkCudaErrors(error);


	free_matrix(h_matrix);
}

void free_matrix(cuda_matrix *matrix)
{
	assert(matrix != NULL);
	cudaError error;
	//error = cudaFree(matrix->d_reference);
	//checkCudaErrors(error);
	error = cudaFree(matrix->d_matrix);
	checkCudaErrors(error);
	error = cudaFree(matrix->d_maxval);
	checkCudaErrors(error);
	error = cudaFree(matrix->d_minval);
	checkCudaErrors(error);
	//error = cudaFree(matrix->d_solution);
	//checkCudaErrors(error);

	free(matrix);
}

//not used
void gen_reference(cuda_matrix *matrix, double *h_matrix, int length)
{
	cudaError error;

	error = cudaMalloc(&(matrix->d_reference), sizeof(double)*(length+1));
	checkCudaErrors(error);
	error = cudaMemcpy(matrix->d_reference, h_matrix, sizeof(double)*(length+1), cudaMemcpyHostToDevice);
	checkCudaErrors(error);
}

void free_matrix(double *matrix)
{
	assert(matrix != NULL);
	free(matrix);
}

void print_matrix(double *matrix, int length)
{
	assert(matrix != NULL);

	/* image row */
	for (int i = 0; i < length; i++){
		printf("%.1f ", (matrix[i] == -0.0)? 0.0 : matrix[i]);
	}
	printf("\n");
}

//not used
__global__ void print_ref(double *ref, int len)
{
	for (int i = 0; i < len+1; ++i)
	{
		printf("%.1f ", ref[i]);
	}
	printf("\n");
}

void print_dev_info()
{
	cudaDeviceProp prop;

	int count;
	checkCudaErrors(cudaGetDevice(&count));
	for (int i = 0; i < 1; ++i)
	{
		checkCudaErrors(cudaGetDeviceProperties(&prop, i));

		printf("----CUDA-DEVICE----\n");
		printf("Name: %s\n", prop.name);
		printf("Clock Rate: %d\n", prop.clockRate);
		printf("Memory: %zuMB\n", prop.totalGlobalMem/1000000);
		printf("Multiprocessors: %d\n", prop.multiProcessorCount);
		printf("Threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Shared Mem per block: %zu\n", prop.sharedMemPerBlock);
		//printf("Gridsize x: %d y: %d\n", prop.maxGridSize[0], prop.maxGridSize[1]);

		printf("\n");
	}
}
