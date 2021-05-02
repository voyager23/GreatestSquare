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

__global__ void func_g(long* d_squares, const long n_squares, long *d_sums, long N, long page_size, int page_idx) {
	// Calc the index of result in device results page
	long i = threadIdx.x + (blockDim.x * blockIdx.x);
	// Calc actual target
	long target = i + (page_size * page_idx);

	if(target <= N) {
		//printf("idx: %ld	target: %ld\n", i, target); return;
		// scan in reverse the squares array
		// save first square which divides i in results[i]
		if(target > 3) {
			for(long x = n_squares-1; x > 0; x -= 1) {
				if((target % d_squares[x]) == 0) {
					d_sums[i] = d_squares[x];
					printf("x: %d target: %ld square: %ld\n", x, target, d_squares[x]);
					break;
				}
			} // for...
		} else {
			d_sums[i] = i;
		}
	} // if target...
}

int main(int argc, char **argv)
{
	cudaError_t error_id;
	long *d_squares = NULL;
	
	// These values based on 1.56GiB available on device
	const int PageY = 190000;
	const int PageX = 1024;
	const int PageSize = PageX*PageY;
	
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
	printf("\nGenerating squares\n");
	set_squares<<<1,limit>>>(d_squares, limit);
	cudaDeviceSynchronize();

#if(DEBUG)	
		// allocate space on host and copy device squares
		long *h_squares = (long *)malloc(sizeof(long )*limit);
		cudaMemcpy(h_squares, d_squares, sizeof(long )*limit, cudaMemcpyDeviceToHost);
		// print long array of squares
		for(long x = 0; x < limit; ++x) printf("%d:%ld  ", x, h_squares[x]); printf("\n");
		// clear host array
		free(h_squares);
#endif
	
	// Allocate a results page on device
	error_id = cudaMalloc(&d_sums, sizeof(long )*PageSize);
	if(error_id != cudaSuccess) {
		printf("cudaMalloc d_sums failed with %d\n", error_id);
		exit(1);
	}		
	// Allocate a results page on host	
	h_sums = (long*)malloc(sizeof(long)*PageSize);
	if(h_sums == NULL) {
		printf("Failed to malloc h_sums.");
		exit(1);
	}
	
	// initialise to zero
	for(int x = 0; x < PageSize; ++x) h_sums[x] = 0L;
	
	int rows = (N / 1024) + 1;
	int pages = (rows / 190000) + 1;
		printf("N: %ld	rows: %d	pages: %d\n", N, rows, pages);
	long Sum = 0;
	long counted = 0;
	for(int pg = 0; pg < pages; ++pg) {
		// launch kernel with appropriate parameters
		
		func_g<<<rows,1024>>>(d_squares, limit, d_sums, N, PageSize, pg);				
		// device sync and test for errors
		error_id = cudaDeviceSynchronize();
		if(error_id != cudaSuccess) {
			printf("cudaDeviceSync returned %d\n", error_id);
			exit(0);
		}
		// copy device sums to host
		error_id = cudaMemcpy(h_sums, d_sums, sizeof(long)*PageSize, cudaMemcpyDeviceToHost);
		
		// DEBUG
		for(int x = 0; ((x<PageSize)&&(counted < N)); ++x,++counted) {
			printf("%d:%ld  ",x,h_sums[x]);
		}
		printf("\n"); goto exit;
		// END DEBUG
		
#if(0)
		// Update S by summing last returned page page
		for(int x = 0; ((x<20)&&(counted < N)); ++x,++counted) {
			Sum += h_sums[x];
		}
#endif
	}

exit:
	// Output Result as S(N) = S
	printf("S(%ld) = %ld.\n", N, Sum);
		
	// CleanUp
	free(h_sums);
	cudaFree(d_sums);
	return 0;
}

