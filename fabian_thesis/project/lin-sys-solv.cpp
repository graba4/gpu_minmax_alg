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
	io_info info = {0, 0, 0, 0, 0, 0, NULL};

	process_args(argc, argv, &info);
	double dur[info.revisions];
	info.durations = dur;
	//print_dev_info();

	/*printf("vars: %d\n", info.v_opt);
	printf("cores: %d\n", info.c_opt);
	printf("implementation: %d\n", info.i_opt);
	printf("revisions: %d\n", info.revisions);*/

	for (info.run_nr = 0; info.run_nr < info.revisions; ++info.run_nr)
	{
		cuda_matrix *matrix = allocate_recources(&info);
		switch(info.i_opt){
			case 0:
				if (info.c_opt == -1)
					info.durations[info.run_nr] = naive_aproach_single_cpu(&info);
					//info.durations[info.run_nr] = naive_aproach_single(matrix);
				else
					info.durations[info.run_nr] = naive_aproach(matrix);

				break;

			case 1:
				if (info.c_opt == -1)
					info.durations[info.run_nr] = naive_aproach_single_cpu(&info);
					//info.durations[info.run_nr] = naive_aproach_single(matrix);
				else
					info.durations[info.run_nr] = naive_aproach_decrement(matrix);

				break;

			case 2:
				if (info.c_opt == -1)
					info.durations[info.run_nr] = naive_aproach_single_cpu(&info);
					//info.durations[info.run_nr] = naive_aproach_single(matrix);
				else
					info.durations[info.run_nr] = naive_aproach_pipe(matrix);

				break;
			
			case 3:
				if (info.c_opt == -1)
					info.durations[info.run_nr] = naive_aproach_single_cpu(&info);
					//info.durations[info.run_nr] = naive_aproach_single(matrix);
				else
					info.durations[info.run_nr] = naive_aproach_decrement_pipe(matrix);

				break;

			case 4:
				if (info.c_opt == -1)
					info.durations[info.run_nr] = naive_aproach_single_cpu(&info);
					//info.durations[info.run_nr] = naive_aproach_single(matrix);
				else
					info.durations[info.run_nr] = naive_aproach_decrement_new(matrix);

				break;

			default:
				error_exit(2, (char *)"Invalid implemetation Nr.");
		}
		free_matrix(matrix);
	}

	process_output(&info);

	return 0;
}