/*
 * eSumSquares.cu
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

#define NL printf("\n")

#define DEBUG 1
//----------------------------------------------------------------------
__global__ void set_squares(long *d_squares, long n_squares) {
	long i = threadIdx.x + (blockIdx.x * blockDim.x);
	if(i < n_squares) d_squares[i] = (int)(i+1)*(i+1);
}

__global__ void func_g(long *managed_sums, long N, long* d_squares, long nSquares) {
	long i = threadIdx.x + (blockDim.x * blockIdx.x);
	if((i == 0)||(i > N)) {
		return;
	}else if(i < 4) {
		managed_sums[i] = 1;
		return;
	} else {
		// search for largest square which divides i
		for(int d = nSquares-1; d >= 0; --d) {
			if((i % d_squares[d]) == 0) {
				managed_sums[i] = d_squares[d];
				return;
			} // if...
		} //for d...
	} // else...
}

//----------------------------------------------------------------------
int main(int argc, char **argv)
{
	// This version will compute s(N) for N<1e8
	const long MaxN = 1e8;
	cudaError_t error_id;
	long *d_squares = NULL;
	
	// extract target N
	long x = 0;
	if(argc == 2) {
		x = atol(argv[1]);
	} else {
		printf("usage: css target (< 1e9)\n");
		exit(1);
	}	
	const long N = x;
	if(N <= MaxN) {
		//printf("target: %ld\n", N);
	} else {
		printf("target: %ld exceeds program limitations %ld\n", N, MaxN);
		exit(2);
	}
	// determine array dimensions for squares
	const long nSquares = (long)(sqrt(N+1));	// defines size of array	

#if(DEBUG)
		printf("target: %ld	nSquares: %ld\n", N, nSquares);
#endif

	// Allocate space on device
	error_id = cudaMalloc(&d_squares, sizeof(long )*nSquares);
	if(error_id != cudaSuccess) {
		printf("cudaMalloc squares failed with %d\n", error_id);
		exit(1);
	}	
	// launch the generator on kernel
	printf("Generating squares\n");
	set_squares<<<1,nSquares>>>(d_squares, nSquares);
	cudaDeviceSynchronize();

#if(DEBUG)	
		// allocate space on host and copy device squares
		long *h_squares = (long *)malloc(sizeof(long )*nSquares);
		cudaMemcpy(h_squares, d_squares, sizeof(long )*nSquares, cudaMemcpyDeviceToHost);
		// print long array of squares
		for(long x = 0; x < nSquares; ++x) printf("%d:%ld  ", x, h_squares[x]); printf("\n");
		// clear host array
		free(h_squares);
#endif

	// allocate managed memory based on N
	const int thdsperblk = 1024;
	const int maxblocks = 1e5;
	const int nblocks = (N / thdsperblk) + 1;
	if (nblocks > maxblocks) {
		printf("%d blocks > maxblocks %d\n", nblocks, maxblocks);
		exit(1);
	}
	long *managed_sums = NULL;
	error_id = cudaMallocManaged(&managed_sums, sizeof(long)*nblocks*thdsperblk);
	if(error_id != cudaSuccess) {
		printf("cudaMallocManaged sums failed with %d\n", error_id);
		exit(1);
	}
	printf("Managed memory: %d blocks of %d threads.\n", nblocks, thdsperblk);
		
	// launch a kernel using calculated configuration
	func_g<<<nblocks, thdsperblk>>>(managed_sums, N, d_squares, nSquares);
	cudaDeviceSynchronize();

	long S = 0;
	// Sum the contents of managed_sums
	for(int s = 1; s <= N; ++s) {
		//printf("sums[%d] = %ld  ", s, managed_sums[s]);
		S += managed_sums[s];
	}
	NL;printf("S(%ld) = %d\n", N, S);
	
	// cleanup code
	cudaFree(d_squares);
	cudaFree(managed_sums);
	return 0;
}

