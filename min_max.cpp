#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdlib.h>
#include "io.h"
#include "errors.h"
#include "implementations.h"
#include "implementations_cpu.h"
#include <assert.h>
#include "resources.h"

int main(int argc, char *argv[])
{
	io_info info; // = {0, 0, 0, 0, 0, 0, 0, NULL};

	process_args(argc, argv, &info);
	double dur[info.revisions];
	info.durations = dur;
	print_dev_info();

	/*
	printf("arrlen: %d\n", info.v_opt);
	printf("SM cores: %d\n", info.c_opt);
	printf("cuda cores: %d\n", info.t_opt);
	printf("implementation: %d\n", info.i_opt);
	printf("revisions: %d\n", info.revisions);
	*/
	for (info.run_nr = 0; info.run_nr < info.revisions; ++info.run_nr)
	{
		cuda_matrix *matrix;	
		switch(info.i_opt){
			case 0:
				matrix = allocate_recources(&info);
				//cuda_matrix *matrix = allocate_recources(&info);
				info.durations[info.run_nr] = cuda_parallel_approach(matrix);
				free_matrix(matrix);
				break;

			case 1:
				matrix = allocate_recources(&info);
				//cuda_matrix *matrix = allocate_recources(&info);
				info.durations[info.run_nr] = sequential_approach(matrix);
				free_matrix(matrix);
				break;

			case 2:
				info.durations[info.run_nr] = thrust_approach(&info);
				break;
			
			case 3:
				matrix = allocate_recources(&info);
				//cuda_matrix *matrix = allocate_recources(&info);
				info.durations[info.run_nr] = thrust_approach_amar(matrix, &info);
				free_matrix(matrix);
				break;

			case 4:
				break;

			default:
				error_exit(2, (char *)"Invalid implemetation Nr.");
		}
		min_max_cpu(matrix);
		assert(verify(matrix));

		free_matrix(matrix);
	}

	process_output(&info);

	return 0;
}
