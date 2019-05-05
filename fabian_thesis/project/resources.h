#ifndef RESOURCES_H
#define RESOURCES_H
#include "io.h"

typedef struct cuda_matrix
{
	double *d_matrix,
		   *d_reference,
		   *d_solution,
		   *d_ref_matrix;
	size_t pitch, pitch2;
	int var_count,
		core_count,
		thread_count;
}cuda_matrix;

cuda_matrix* allocate_recources(io_info *info);
void free_matrix(cuda_matrix *matrix);
void print_dev_info();

#endif