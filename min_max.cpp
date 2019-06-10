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
	double gpu_dur[info.revisions];
	info.durations_gpu = gpu_dur;
	double cpu_dur[info.revisions];
	info.durations_cpu = cpu_dur;
	for (int i = 0; i < info.revisions; ++i)
		info.durations_gpu[i] = info.durations_cpu[i] = 0;

	//print_dev_info();

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
			case 0: //cuda parallel
				matrix = allocate_recources(&info, info.run_nr);
				//cuda_matrix *matrix = allocate_recources(&info, info.run_nr);
				info.durations_gpu[info.run_nr] += cuda_parallel_approach(matrix);
				break;

			case 1: //cuda sequential
				matrix = allocate_recources(&info, info.run_nr);
				//cuda_matrix *matrix = allocate_recources(&info, info.run_nr);
				info.durations_gpu[info.run_nr] += sequential_approach(matrix);
				break;

			case 2: //thrust
				info.durations_gpu[info.run_nr] += thrust_approach(&info);
				break;
			
			case 3: //thrust
				matrix = allocate_recources(&info, info.run_nr);
				//cuda_matrix *matrix = allocate_recources(&info, info.run_nr);
				info.durations_gpu[info.run_nr] += thrust_approach_amar(matrix, &info);
				break;

			case 4:
				break;

			default:
				error_exit(2, (char *)"Invalid implemetation Nr.");
		}
		info.durations_cpu[info.run_nr] += min_max_cpu(matrix);
		assert(verify(matrix));

		free_matrix(matrix);
	}

	process_output(&info);

	return 0;
}
