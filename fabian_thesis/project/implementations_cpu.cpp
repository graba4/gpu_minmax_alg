#include "implementations_cpu.h"
#include "io.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>

double **create_matrix(int var_cnt, bool clear);
void divide_cpu(double *arr, int length, int index);
void free_matrix(double **matrix, int length);
void print_matrix(double **matrix, int length);
bool check_solutions(double *line, double *solution, int length);
double *gen_reference(double **matrix, int length);

double naive_aproach_single_cpu(io_info *info){
	int vars = info->v_opt;
	double **matrix = create_matrix(vars, false);
	double *solution = (double*)malloc(sizeof(double)*vars),
		   *reference = gen_reference(matrix, vars);
	assert(solution != NULL);
	//print_matrix(matrix, vars);

	clock_t begin = clock();

	for (int i = 0; i < vars; ++i)
	{
		divide_cpu(matrix[i], vars+1, i);
		
		for (int y = i; y < vars-1; ++y)
		{
			double factor = matrix[y+1][i];
			if (factor == 0)
				continue;

			for (int x = 0; x < vars+1; ++x)
			{
				matrix[y+1][x] -= matrix[i][x] * factor;
			}
		}
	}

	for (int y = vars-1; y >= 0; --y)
	{
		double temp = matrix[y][vars];
		int x = vars-1;
		while(matrix[y][x] != 1.0){
			assert(x > 0);
			temp -= matrix[y][x]*solution[x];
			x--;
		}

		solution[x] = temp;
	}

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	//print_matrix(matrix, vars);
	assert(check_solutions(reference, solution, vars));

	free_matrix(matrix, vars);
	free(solution);

	return time_spent;
}

void divide_cpu(double *arr, int length, int index){
	assert((index < length) && (index >= 0));
	double div = arr[index];

	for (int i = index; i < length; ++i)
	{
		arr[i] /= div;
	}
}

#define DEC_COUNT (1000)
double **create_matrix(int var_cnt, bool clear)
{
	double **matrix = (double **)malloc(sizeof(double *) * var_cnt);
	assert(matrix != NULL);

	if (clear) {
		for (int y = 0; y < var_cnt; y++) {
			matrix[y] = (double *)calloc((var_cnt+1), sizeof(double));
			assert(matrix[y] != NULL);
		}
	} else {
		for (int y = 0; y < var_cnt; y++) {
			matrix[y] = (double *)malloc(sizeof(double) * (var_cnt+1));
			assert(matrix[y] != NULL);
			for (int i = 0; i < var_cnt+1; ++i){
				srand(time(NULL)*(i+1)*(y+1));
				matrix[y][i] = ((double)rand()/(double)RAND_MAX)*DEC_COUNT;
			}
		}
	}

	return matrix;
}

void free_matrix(double **matrix, int length)
{
	assert(matrix != NULL);

	for (int y = 0; y < length; y++)
		free(matrix[y]);

	free(matrix);
}

void print_matrix(double **matrix, int length)
{
	assert(matrix != NULL);

	for (int j = 0; j < length; j++) {
		/* image row */
		for (int i = 0; i < length+1; i++){
			if (i == length)
				printf("|%.1f", (matrix[j][i] == -0.0)? 0.0 : matrix[j][i]);
			else
				printf("%.1f ", (matrix[j][i] == -0.0)? 0.0 : matrix[j][i]);
		}
		printf("\n");
	}
	printf("\n");
}

#define DEVIATION (0.001) //in %
bool check_solutions(double *line, double *solution, int length){
	double sum = 0, dev;
	for (int i = 0; i < length; ++i)
	{
		sum += line[i] * solution[i];
	}

	//printf("sum: %.25f line: %.25f\n", sum, line[length]);
	dev = ((line[length] - sum)*100)*sum;
	dev = (dev < 0) ? dev*(-1) : dev;
	//printf("%.30f\n", dev);
	return dev < DEVIATION;
}

double *gen_reference(double **matrix, int length){
	double *reference = (double*)malloc(sizeof(double)*(length+1));

	for (int i = 0; i < length+1; ++i)
	{
		reference[i] = matrix[0][i];
	}

	return reference;
}