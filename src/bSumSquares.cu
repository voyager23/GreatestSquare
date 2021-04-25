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
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
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
	
	ulong N = 1024*1024;	// sum perfect square divisors up to this value 
		
	// Allocate and set the host 'perfect squares' array
	ulong root_max = (ulong)floor(sqrt((double)N));	
	const ulong n_squares = root_max + 1;	
	ulong h_squares[n_squares];
	for(int x = 0; x < n_squares; x += 1) h_squares[x] = x*x;
	// Allocate memory on device for 'squares'
	ulong *d_squares;
	error_id = cudaMalloc((void**)&d_squares, sizeof(ulong)*n_squares);
	if(error_id != cudaSuccess) {
		printf("cudaMalloc squares failed with %d\n", error_id);
		exit(1);
	}
	// Copy squares to device
	error_id = cudaMemcpy(d_squares, h_squares, sizeof(ulong)*n_squares,
		cudaMemcpyHostToDevice);
	if(error_id != cudaSuccess) {
		printf("cudaMemcpy squares to device failed with %d\n", error_id);
		exit(1);
	}

	// Allocate memory on host and device for 2 pages of N results
	ulong *results_0 = NULL, *results_1 = NULL;
	error_id = cudaMallocManaged((void**)results_0, sizeof(ulong)*(N+1));
	if(error_id != cudaSuccess) {
		printf("cudaMallocManaged (0) failed with %d\n", error_id);
		exit(1);
	}
	error_id = cudaMallocManaged((void**)results_1, sizeof(ulong)*(N+1));
	if(error_id != cudaSuccess) {
		printf("cudaMallocManaged (1) results failed with %d\n", error_id);
		exit(1);
	}
	
	// Set variables
	ulong total = 0;
	for(x = 0; x <= N; ++x) {
		results_0 = 0;
	}							// clear results
	ulong *pagePtr = results_1;	// new results go here
	ulong pageIdx = 0;			// page counter
	
	// set configuration
	dim3 thread_size = (1024,1,1);
	dim3 block_size = (1024,1,1);
	
	
	
	
	// launch kernel
	// kernel<<<grid_size, block_size>>>(d_squares, n_squares, d_results, (N+1));
	
	// Wait for device to finish?
	//cudaDeviceSynchronize();
	
	
	// Print results array
	// for(int x = 0; x < N+1; ++x) printf("%d:%ld  ", x, h_results[x]);
	// printf("\n");

	// Cleanup
	cudaFree(d_squares);
	cudaFree(results_0);
	cudaFree(results_1);
	
	return 0;
}

