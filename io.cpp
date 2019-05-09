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

#define STRLEN (1024)
#define BETWEEN(value, min, max) (value <= max && value >= min)

char* usage(char *argv[], char *usage_msg){
	(void)sprintf(usage_msg, "Usage: %s -v arrlen -w windowsize -c cores -i implementation [-r revisions] [-f file] [-t threads]", argv[0]);
	return usage_msg;
}

void process_args(int argc, char *argv[], io_info *info){
	int opt,
		opt_int,
		opt_cnt = 1,
		revisions_opt = 1;

	bool v_opt = false,
		 c_opt = false,
		 i_opt = false,
		 t_opt = false,
		 w_opt = false;
	char *endptr;
	char usage_str[STRLEN];

	info->f = stdout;

	while ((opt = getopt(argc, argv, "v:c:i:w:r:f:t:")) != -1) {
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

		    case 'w':
		    	opt_int = strtol(optarg, &endptr, 10);
			    if (*endptr != '\0')
			    	error_exit(1, usage(argv, usage_str));
			    if (opt_int < 1)
			    {
			    	(void)fprintf(stderr, "Invalid number of windowsize!\n");
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

	if (!t_opt) 
		info->t_opt = -1;

	info->revisions = revisions_opt;
}

void process_output(io_info *info){
	double average = 0;
	for (int i = 0; i < info->revisions; ++i)
	{
		average += info->durations[i];
	}
	average = average/info->revisions;

	fprintf(info->f, "%d,%d,%d,%.2f,%d\n", info->v_opt, info->w_opt, info->c_opt, average, info->i_opt);
	fflush(info->f);
	fclose(info->f);
}