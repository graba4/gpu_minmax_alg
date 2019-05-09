#ifndef CUDA_DEQUE_H
#define CUDA_DEQUE_H

typedef struct int_node
{
	int val;
	int_node *next;
	int_node *prev;
}int_node;

class cuda_deque
{
public:
	__device__ cuda_deque();
	__device__ ~cuda_deque();
	__device__ uint size();
	__device__ uint front();
	__device__ uint back();
	__device__ void push_back(int val);
	__device__ void push_front(int val);
	__device__ void pop_front();
	__device__ void pop_back();
	__device__ void print();

private:
	uint len;
	int_node *first;
	int_node *last;
};


#endif