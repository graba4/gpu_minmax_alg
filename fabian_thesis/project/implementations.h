#ifndef IMPLEMENTATIONS_H
#define IMPLEMENTATIONS_H
#include "resources.h"

double naive_aproach_decrement(cuda_matrix *matrix);
double naive_aproach_decrement_new(cuda_matrix *matrix);
double naive_aproach_decrement_pipe(cuda_matrix *matrix);
double naive_aproach(cuda_matrix *matrix);
double naive_aproach_pipe(cuda_matrix *matrix);
#endif