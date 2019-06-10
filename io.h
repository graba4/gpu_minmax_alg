#ifndef IO_H
#define IO_H
#include <stdio.h>

typedef struct io_info
{
	unsigned int v_opt,
	c_opt,
	i_opt,
	t_opt,
	w_opt,
	a_opt,
	revisions,
	seed,
	run_nr;
	FILE *f;

	double *durations_gpu;
	double *durations_cpu;
} io_info;

void process_args(int argc, char *argv[], io_info *info);
void process_output(io_info *info);

#endif