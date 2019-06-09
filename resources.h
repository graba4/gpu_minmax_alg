#ifndef RESOURCES_H
#define RESOURCES_H
#include "io.h"

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

cuda_matrix* allocate_recources(io_info *info);
void free_matrix(cuda_matrix *matrix);
void print_dev_info();
void print_matrix(double *matrix, int length);

#endif