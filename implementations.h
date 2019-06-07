#ifndef IMPLEMENTATIONS_H
#define IMPLEMENTATIONS_H
#include "resources.h"

double cuda_parallel_approach(cuda_matrix *matrix);
double sequential_approach(cuda_matrix *matrix);
double thrust_approach(io_info *info);
double thrust_approach_amar(cuda_matrix *matrix, io_info *info);

#endif