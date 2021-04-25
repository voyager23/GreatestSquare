/*
 * bSumSquares.cu
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

__global__ void kernel(ulong* d_squares, const ulong n_squares, ulong* d_results, ulong N) {
	ulong i = threadIdx.x + (blockIdx.x * blockDim.x);
	if(i < N) {
		// scan in reverse the squares array
		// save first square which divides i in results[i]
		if(i > 3) {
			for(int x = n_squares-1; x > 0; x -= 1) {
				if((i % d_squares[x]) == 0) {
					d_results[i] = d_squares[x];
					break;
				}
			} // for...
		} else {
			d_results[i] = i;
		}
	} //
}

int main(int argc, char **argv)
{
	cudaError_t error_id;
	
	// extract target N
	ulong x = 0;
	if(argc == 2) {
		x = atol(argv[1]);
	} else {
		printf("usage: css target (< 10^9)\n");
		exit(1);
	}	
	const ulong N = x;
	if(N <= 1000000000L) {
		printf("target: %ld\n", N);
	} else {
		printf("target: %ld exceeds program limitations\n", N);
		exit(2);
	}
	// calc. and copy table of perfect squares to device
	// determine array dimensions
	double limit = sqrt(sqrt(N)) + 1;
	printf("target: %ld		limit: %ld\n", N, (ulong)limit);

	//-------
	return 0;
}

