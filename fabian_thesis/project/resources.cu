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
void create_matrix(cuda_matrix *matrix, int var_cnt, bool clear);
void print_matrix(double *matrix, int length);
void gen_reference(cuda_matrix *matrix, double *h_matrix, int length);
void print_dev_info();

cuda_matrix* allocate_recources(io_info *info)
{
	cudaError error;
	int vars = info->v_opt;

	cuda_matrix *matrix = (cuda_matrix*)malloc(sizeof(cuda_matrix));
	assert(matrix != NULL);
	matrix->var_count = vars;
	matrix->core_count = info->c_opt;
	matrix->thread_count = info->t_opt;

	create_matrix(matrix, vars, false);

	error = cudaMalloc(&(matrix->d_solution), sizeof(double)*vars);
	checkCudaErrors(error);

	return matrix;
}

#define DEC_COUNT (1000)
void create_matrix(cuda_matrix *matrix, int var_cnt, bool clear)
{
	cudaError error;
	double *h_matrix;

	if (clear) {
		h_matrix = (double *)calloc((var_cnt+1) * var_cnt, sizeof(double));
		assert(h_matrix != NULL);
	} else {
		h_matrix = (double *)malloc(sizeof(double) * (var_cnt+1) * var_cnt);
		assert(h_matrix != NULL);
		unsigned int seed = time(NULL);

		for (int y = 0; y < var_cnt; y++) {
			for (int i = 0; i < var_cnt+1; ++i){
				srand(seed*(y+1)*(var_cnt+1)+i);
				h_matrix[y*(var_cnt+1)+i] = ((double)rand()/(double)RAND_MAX)*DEC_COUNT;
				assert(h_matrix[y*(var_cnt+1)+i] > 0);
				if(h_matrix[y*(var_cnt+1)+i] == 0)
					h_matrix[y*(var_cnt+1)+i] = 1;
			}
		}
	}

	error = cudaMallocPitch(&(matrix->d_matrix), &(matrix->pitch),
			sizeof(double)*(var_cnt+1), var_cnt);
	checkCudaErrors(error);

	error = cudaMemcpy2D(matrix->d_matrix, matrix->pitch, h_matrix,
			sizeof(double)*(var_cnt+1), sizeof(double)*(var_cnt+1), var_cnt, cudaMemcpyHostToDevice);
	checkCudaErrors(error);

	gen_reference(matrix, h_matrix, var_cnt);
	
	free_matrix(h_matrix);
}

void free_matrix(cuda_matrix *matrix)
{
	assert(matrix != NULL);
	cudaError error;
	error = cudaFree(matrix->d_reference);
	checkCudaErrors(error);
	error = cudaFree(matrix->d_matrix);
	checkCudaErrors(error);
	error = cudaFree(matrix->d_solution);
	checkCudaErrors(error);

	free(matrix);
}

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

	for (int j = 0; j < length; j++) {
		/* image row */
		for (int i = 0; i < length+1; i++){
			if (i == length)
				printf("|%.1f", (matrix[j*(length+1)+i] == -0.0)? 0.0 : matrix[j*(length+1)+i]);
			else
				printf("%.1f ", (matrix[j*(length+1)+i] == -0.0)? 0.0 : matrix[j*(length+1)+i]);
		}
		printf("\n");
	}
	printf("\n");
}

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