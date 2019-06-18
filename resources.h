#ifndef RESOURCES_H
#define RESOURCES_H
#include "io.h"

#define CHUNK_SIZE (5)


typedef struct cuda_matrix
{
	double *d_matrix, //the array where we want to operate
		   *d_maxval,
		   *d_minval,
		   *h_maxval,
		   *h_minval,
		   *h_matrix;
	unsigned int seed;
	int arrlen, //length of d_matrix
		window_size,
		core_count, //streaming multiproc
		thread_count; //cuda cores
}cuda_matrix;

cuda_matrix* allocate_resources(io_info *info, int run_nr);
cuda_matrix* allocate_resources_streams(io_info *info, int run_nr);
void free_matrix(cuda_matrix *matrix);
void print_dev_info();
void print_matrix(double *matrix, int length);

#endif