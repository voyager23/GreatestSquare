/*
 * cSumSquares.cu
 * 
 * Copyright 2021 mike <mike@fedora33>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * =====================================================================
 * Function g(n) is defined as greatest perfect square which divides n.
 * Consider n = 10^9. 31623^2 = 1000014129
 * To build a table of useful perfect squares: for x in range(1, 31624) calc x^2
 * Requires 976577 blocks of 1024 threads
 * Host memory approx 14Gb free, 10^9
 * 
 * =====================================================================
 */

#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define DEBUG 1
#define MANAGED 0

__global__ void set_squares(long *d_squares, long n_squares) {
	long i = threadIdx.x + (blockIdx.x * blockDim.x);
	if(i < n_squares) d_squares[i] = (int)(i+1)*(i+1);
}

__global__ void func_g(int* d_squares, const long limit, long *h_sums, long N) {
	long i = threadIdx.x + (blockDim.x * blockIdx.x);
	if(i < N) {
		// scan in reverse the squares array
		// save first square which divides i in results[i]
		if(i > 3) {
			for(long x = limit-1; x > 0; x -= 1) {
				if((i % d_squares[x]) == 0) {
					h_sums[i] = d_squares[x];
					break;
				}
			} // for...
		} else {
			h_sums[i] = i;
		}
	} //
}

int main(int argc, char **argv)
{
	cudaError_t error_id;
	long *d_squares = NULL;
	long *h_sums = NULL;
	
	// extract target N
	long x = 0;
	if(argc == 2) {
		x = atol(argv[1]);
	} else {
		printf("usage: css target (< 1e8)\n");
		exit(1);
	}	
	const long N = x;
	if(N <= 1000000000L) {
		printf("target: %ld\n", N);
	} else {
		printf("target: %ld exceeds program limitations\n", N);
		exit(2);
	}
	// determine array dimensions
	long limit = (long)(sqrt(sqrt(N)) + 1);	// defines size of array	

#if(DEBUG)
		printf("target: %ld		limit: %ld\n", N, limit);
#endif

	// Allocate space on device
	error_id = cudaMalloc(&d_squares, sizeof(long )*limit);
	if(error_id != cudaSuccess) {
		printf("cudaMalloc squares failed with %d\n", error_id);
		exit(1);
	}	
	// launch the generator on kernel
	set_squares<<<1,limit>>>(d_squares, limit);
	cudaDeviceSynchronize();

#if(DEBUG)	
		// allocate space on host and copy device squares
		long *h_squares = (long *)malloc(sizeof(long )*limit);
		cudaMemcpy(h_squares, d_squares, sizeof(long )*limit, cudaMemcpyDeviceToHost);
		// prlong array
		for(long x = 0; x < limit; ++x) printf("%d:%ld  ", x, h_squares[x]); printf("\n");
		// clear host array
		free(h_squares);
#endif

#if(MANAGED)
	// Allocate managed memory for results
	// Note: Fails for more than 1e8 long int
	error_id = cudaMallocManaged(&h_sums, sizeof(long)*N);
	if(error_id != cudaSuccess) {
		printf("cudaMallocManaged sums failed with %d\n", error_id);
		exit(1);
	}
#else
	h_sums = (long*)malloc(sizeof(long)*N);
	if(h_sums == NULL){
		printf("malloc() h_sums failed.\n");
		exit(1);
	}
#endif
	
	
	
	// calculate the launch config based in thread blocks of 1024 threads
	const int nBlocks = (N/1024) + 1;
	// launch the kernel
	//func_g<<<nBlocks,1024>>()
	

	//-------
	cudaFree(d_squares);
	cudaFree(h_sums);
	return 0;
}

