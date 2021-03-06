#include "io.h"
#include <stdbool.h>
#include "errors.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <assert.h>
#include <helper_functions.h>

#define STRLEN (1024)
#define BETWEEN(value, min, max) (value <= max && value >= min)

char* usage(char *argv[], char *usage_msg){
	(void)sprintf(usage_msg, "Usage: %s -v arrlen -w windowsize -c cores -i implementation [-r revisions] [-f file] [-s seed] [-t threads] [-a]", argv[0]);
	return usage_msg;
}

void process_args(int argc, char *argv[], io_info *info){
	unsigned int opt,
		opt_int,
		opt_cnt = 1,
		revisions_opt = 1,
		seed_opt = 0;

	bool v_opt = false,
		 c_opt = false,
		 i_opt = false,
		 t_opt = false,
		 w_opt = false,
		 s_opt = false,
		 a_opt = false;
	char *endptr;
	char usage_str[STRLEN];

	info->f = stdout;

	while ((opt = getopt(argc, argv, "av:c:i:w:r:s:f:t:")) != -1) {
		switch (opt) {
		    case 'v':
		    	opt_int = strtol(optarg, &endptr, 10);
			    if (*endptr != '\0')
			    	error_exit(1, usage(argv, usage_str));
			    if (opt_int < 1)
			    {
			    	(void)fprintf(stderr, "Invalid number of variables!\n");
			    	error_exit(1, usage(argv, usage_str));
			    }
		    	opt_cnt+=2;
				v_opt = true;
				info->v_opt = opt_int;
		        break;

		    case 'a':
			    opt_cnt+=1;
				a_opt = true;
		    	break;

		    case 'w':
		    	opt_int = strtol(optarg, &endptr, 10);
			    if (*endptr != '\0')
			    	error_exit(1, usage(argv, usage_str));
			    if (opt_int < 3)
			    {
			    	(void)fprintf(stderr, "Invalid number of windowsize! Minimum 3\n");
			    	error_exit(1, usage(argv, usage_str));
			    }
		    	opt_cnt+=2;
				w_opt = true;
				info->w_opt = opt_int;
		        break;

			case 'c':
				opt_int = strtol(optarg, &endptr, 10);
				if (*endptr != '\0')
					error_exit(1, usage(argv, usage_str));
				if ((opt_int < 1) && (opt_int != -1))
				{
					(void)fprintf(stderr, "Invalid number of cores!\n");
					error_exit(1, usage(argv, usage_str));
				}
				opt_cnt+=2;
				c_opt = true;
				info->c_opt = opt_int;
		        break;

		    case 'i':
		    	opt_int = strtol(optarg, &endptr, 10);
			    if (*endptr != '\0')
					error_exit(1, usage(argv, usage_str));
		    	opt_cnt+=2;
		    	i_opt = true;
		    	info->i_opt = opt_int;
		  		break;

		  	case 'r':
		  		revisions_opt = strtol(optarg, &endptr, 10);
			    if (*endptr != '\0')
					error_exit(1, usage(argv, usage_str));
				if (revisions_opt < 1)
				{
					(void)fprintf(stderr, "Invalid number of revisions!\n");
					error_exit(1, usage(argv, usage_str));
				}
		    	opt_cnt+=2;
		  		break;

		  	case 's':
		  		seed_opt = strtol(optarg, &endptr, 10);
			    if (*endptr != '\0')
					error_exit(1, usage(argv, usage_str));
				if (seed_opt == 0)
				{
					(void)fprintf(stderr, "seed 0 is not allowed\n");
					error_exit(1, usage(argv, usage_str));
				}
				opt_cnt+=2;
				s_opt = true;
		  		break;

		  	case 'f':
		  		info->f = fopen(optarg, "a+");
		  		opt_cnt+=2;
		  		assert(info->f != NULL);
		  		break;

		  	case 't':
		  		opt_int = strtol(optarg, &endptr, 10);
				if (*endptr != '\0')
					error_exit(1, usage(argv, usage_str));
				if ((opt_int < 1) && (opt_int != -1))
				{
					(void)fprintf(stderr, "Invalid number of cores!\n");
					error_exit(1, usage(argv, usage_str));
				}
				opt_cnt+=2;
				t_opt = true;
				info->t_opt = opt_int;
		        break;

		    default:
		      	error_exit(1, usage(argv, usage_str));
	    }
	}

	if (!(v_opt && w_opt && c_opt && i_opt) || (argc != opt_cnt))
		error_exit(1, usage(argv, usage_str));

	if (info->w_opt > info->v_opt){
		sprintf(usage_str, "windowsize too large");
		error_exit(1, usage_str);
	}

	info->a_opt = a_opt;

	if (!t_opt) 
		info->t_opt = 1;

	info->seed = seed_opt;
	info->revisions = revisions_opt;
}

void process_output(io_info *info){
	double average_gpu = 0,
		   average_cpu = 0;
	for (int i = 0; i < info->revisions; ++i)
	{
		average_gpu += info->durations_gpu[i];
		//printf("gpu time: %.2f\n", info->durations_gpu[i]);
		average_cpu += info->durations_cpu[i];
		//printf("cpu time: %.2f\n", info->durations_cpu[i]);

	}
	average_gpu = average_gpu/info->revisions;
	average_cpu = average_cpu/info->revisions;

	printf("Cuda took: %f seconds\n", average_gpu);
	printf("CPU took: %f seconds\n", average_cpu);

	fprintf(info->f, "%d,%d,%d,%.2f,%.2f,%d,%d\n", info->v_opt, info->w_opt, info->c_opt, average_gpu, average_cpu, info->i_opt, info->seed);
	fflush(info->f);
	fclose(info->f);
}
