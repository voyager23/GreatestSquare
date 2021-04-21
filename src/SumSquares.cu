/*
 * SumSquares.cu
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
#include <driver_types.h>	// cudaError_t
#include <vector_types.h>
#include <cuda_runtime.h>	// cudaMalloc, cudaFree

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
	
	// Allocate and set the host 'squares' array
	ulong N = 1025;	
	ulong root_max = (ulong)floor(sqrt((double)N));	
	const ulong n_squares = root_max + 1;	
	ulong h_squares[n_squares];
	for(int x = 0; x < n_squares; x += 1) h_squares[x] = x*x;
	
	// Allocate host results array
	ulong h_results[N+1];
	
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
	// Allocate memory on device for N results
	ulong *d_results;
	error_id = cudaMalloc((void**)&d_results, sizeof(ulong)*(N+1));
	if(error_id != cudaSuccess) {
		printf("cudaMalloc results failed with %d\n", error_id);
		exit(1);
	}
	
	// Set configuration parameters
	const ulong Nthreads = 1024;	// max number threads/block
	const ulong Nblocks = (N/Nthreads)+1;
	dim3 grid_size=(Nblocks); dim3 block_size=Nthreads;
	
	// launch kernel
	kernel<<<grid_size, block_size>>>(d_squares, n_squares, d_results, (N+1));
	
	// Wait for device to finish?
	//cudaDeviceSynchronize();
	
	// copy N results back to host
	error_id = cudaMemcpy(h_results, d_results, sizeof(ulong)*(N+1),
		cudaMemcpyDeviceToHost);
	if(error_id != cudaSuccess) {
		printf("cudaMemcpy to host  failed with %d\n", error_id);
		exit(1);
	}
	
	// Print results array
	for(int x = 0; x < N+1; ++x) printf("%d:%ld  ", x, h_results[x]);
	printf("\n");

	// Cleanup
	cudaFree(d_squares);
	cudaFree(d_results);
	
	return 0;
}

