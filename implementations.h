#ifndef IMPLEMENTATIONS_H
#define IMPLEMENTATIONS_H
#include "resources.h"

double cuda_parallel_approach(cuda_matrix *matrix);
double sequential_approach(cuda_matrix *matrix);
double thrust_approach(cuda_matrix *matrix);
double streams_approach(cuda_matrix *matrix);


#endif