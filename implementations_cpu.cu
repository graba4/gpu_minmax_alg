#include "implementations_cpu.h"
#include <deque>
#include <stdio.h>
#include <stdbool.h>
#include "resources.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
//#include <time.h>
#include <ctime>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define MIN_WIN_SIZE (3)
double min_max_cpu(cuda_matrix *matrix){
	assert(matrix != NULL);
	int window_size = matrix->window_size;

	assert(window_size >= MIN_WIN_SIZE);
	std::deque<int> u, l;
	double *a = matrix->h_matrix,
		   *minval = matrix->h_minval,
		   *maxval = matrix->h_maxval;
	//print_matrixx(a, arrlen);
	clock_t begin = clock();

	for(u_int i = 1; i < matrix->arrlen; ++i){
		if(i >= window_size){
			maxval[i-window_size] = a[u.size() > 0 ? u.front():i-1];
			minval[i-window_size] = a[l.size() > 0 ? l.front():i-1];
		}

		if(a[i] > a[i-1]){
			l.push_back(i-1);
			
			if(i == window_size + l.front())
				l.pop_front();

			while(u.size() > 0){
				if(a[i] <= a[u.back()]){
					if(i == window_size + u.front())
						u.pop_front();
					break;
				}
				u.pop_back();
			}
		}else{
			u.push_back(i-1);
			
			if(i == window_size + u.front())
				u.pop_front();
			
			
			while(l.size() > 0){
				if(a[i] >= a[l.back()]){
					if(i == window_size + l.front())
						l.pop_front();
					break;
				}
				l.pop_back();
			}
		}
	}
	maxval[matrix->arrlen - window_size] = a[u.size() > 0 ? u.front() : matrix->arrlen-1];
	minval[matrix->arrlen - window_size] = a[l.size() > 0 ? l.front() : matrix->arrlen-1];	
	
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	return time_spent;
}

bool verify(cuda_matrix *matrix){
	assert(matrix != NULL);
	double *cuda_minval = (double*)malloc(sizeof(double)*(matrix->arrlen)),
		   *cuda_maxval = (double*)malloc(sizeof(double)*(matrix->arrlen));

	assert(cuda_maxval != NULL);
	assert(cuda_minval != NULL);
	assert(matrix->d_minval != NULL);
	assert(matrix->d_maxval != NULL);

	cudaError error;
	error = cudaMemcpy(cuda_minval, matrix->d_minval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
	checkCudaErrors(error);
	error = cudaMemcpy(cuda_maxval, matrix->d_maxval, matrix->arrlen*sizeof(double), cudaMemcpyDeviceToHost);
	checkCudaErrors(error);
	//print_matrix(matrix->h_matrix, matrix->arrlen);
	//print_matrix(cuda_minval, matrix->arrlen);
	//print_matrix(matrix->h_maxval, matrix->arrlen);

	for(int i = 0; i < matrix->arrlen; ++i) {
		if (cuda_minval[i] != matrix->h_minval[i]){
			return false;
		}
		if (cuda_maxval[i] != matrix->h_maxval[i]){
			return false;
		}
	}

	return true;
}