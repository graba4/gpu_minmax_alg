#include "cuda_deque.h"
#include "assert.h"
#include <stdio.h>

__device__ cuda_deque::cuda_deque(){
	len = 0;
	first = NULL;
	last = NULL;
}

__device__ cuda_deque::~cuda_deque(){
	int_node *n = first,
			 *temp;

	while(n != NULL){
		temp = n->next;
		delete n;
		n = temp;
	}
}

__device__ uint cuda_deque::size(){
	return len;
}

__device__ uint cuda_deque::front(){
	assert(first != NULL);
	return first->val;
}

__device__ uint cuda_deque::back(){
	assert(last != NULL);
	return last->val;
}

__device__ void cuda_deque::print(){
	int_node *n = first,
			 *temp;

	printf("print arr:\n");
	while(n != NULL){
		temp = n->next;
		printf("%d ", n->val);
		n = temp;
	}
	printf("\n");

}

__device__ void cuda_deque::push_back(int val){
	int_node *temp = new int_node();
	len++;
	temp->next = NULL;
	temp->prev = last;
	temp->val = val;
	if (last != NULL)
		last->next = temp;
	if (first == NULL)
		first = temp;
	last = temp;
}

__device__ void cuda_deque::push_front(int val){
	//not used
}

__device__ void cuda_deque::pop_front(){
	if (len == 0 || first == NULL)
		return;
	len--;

	int_node *temp = first->next;
	delete first;
	first = temp;
	if (first == NULL)
		last = NULL;
	else
		first->prev = NULL;
}

__device__ void cuda_deque::pop_back(){
	if (len == 0 || last == NULL)
		return;
	len--;

	int_node *temp = last->prev;
	delete last;
	last = temp;
	if (last == NULL)
		first = NULL;
	else
		last->next = NULL;
}