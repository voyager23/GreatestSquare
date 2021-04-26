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
 * 
 * Host memory approx 14Gb free, 10^9
 * 
 * Available device memory approx. 1.6GiB
 * Using a maximum of 190000 * 1024 * sizeof(long) requires 1.56GiB
 * For a given value of N:
 * 	calc lines required (N/1024) + 1
 * 	pagecount = (lines/190000) + 1
 * 
 * Launch the kernel pagecount times, summing partial results
 * 
 * =====================================================================
 */

#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define DEBUG 1

__global__ void set_squares(long *d_squares, long n_squares) {
	long i = threadIdx.x + (blockIdx.x * blockDim.x);
	if(i < n_squares) d_squares[i] = (int)(i+1)*(i+1);
}

//__global__ void func_g(int* d_squares, const long limit, long *h_sums, long N) {
__global__ void func_g() {
	

	return;
	//END DEBUG
#if(0)	
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
#endif
}

int main(int argc, char **argv)
{
	cudaError_t error_id;
	long *d_squares = NULL;
	
	// These values based on 1.56GiB available on device
	const int PageY = 190000;
	const int PageX = 1024;
	int lines = 0;
	int pages = 0;
	
	long *h_sums = NULL;	// large page of partial results
	long *d_sums = NULL;
	
	// extract target N
	long x = 0;
	if(argc == 2) {
		x = atol(argv[1]);
	} else {
		printf("usage: css target (< 1e9)\n");
		exit(1);
	}	
	const long N = x;
	if(N <= 1e9L) {
		printf("target: %ld\n", N);
	} else {
		printf("target: %ld exceeds program limitations (1e9)\n", N);
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
	
	// Allocate a results page on device
	error_id = cudaMalloc(&d_sums, sizeof(long )*PageX*PageY);
	if(error_id != cudaSuccess) {
		printf("cudaMalloc d_sums failed with %d\n", error_id);
		exit(1);
	}		
	// Allocate a results page on host	
	h_sums = (long*)malloc(sizeof(long)*PageX*PageY);
	if(h_sums == NULL) {
		printf("Failed to malloc h_sums.");
		exit(1);
	}
	lines = (N / 1024) + 1;
	pages = (lines / 190000) + 1;
	
	printf("N: %ld	lines: %d	pages: %d\n", N, lines, pages);
	
	// CleanUp
	free(h_sums);
	return 0;
}

