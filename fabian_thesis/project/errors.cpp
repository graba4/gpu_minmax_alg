#include "errors.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * Error handler:
 * 1 Usage-Error
 * 2 Invalid implementation
 * 3 hardware Error
 */

void error_exit(int code, char *msg){
	fprintf(stderr, "%s\n", msg);
	exit(code);
}