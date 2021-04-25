/*
 * dev1.c
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
 */


#include <stdio.h>
#include <math.h>

#include <cuda.h>

__global__ void kernel(ulong* squares, ulong size) {
	ulong i = threadIdx.x + (blockIdx.x * blockDim.x);
	if(i < size) squares[i] *= 10;
}

int main(int argc, char **argv)
{
	cudaError_t error_id;
	
	ulong N = 1000;
	
	ulong root_max = (ulong)floor(sqrt((double)N));
	
	printf("%ld\n",root_max);
	
	const ulong size_squares = root_max + 1;
	
	ulong squares[size_squares];
	for(int x = 0; x < size_squares; x += 1) squares[x] = x*x;
	
	for(int x = 0; x < root_max+1; x += 1) {
		printf("%ld ", squares[x]);
	}
	printf("\n");
	
	// allocate memory on device
	ulong *d_c;
	error_id = cudaMalloc((void**)&d_c, sizeof(ulong)*size_squares);
	if(error_id != cudaSuccess) {
		printf("cudaMalloc failed with %d\n", error_id);
		exit(1);
	}
	
	// copy data to device
	error_id = cudaMemcpy(d_c, squares, sizeof(ulong)*size_squares,
		cudaMemcpyHostToDevice);
	if(error_id != cudaSuccess) {
		printf("cudaMemcpy to device failed with %d\n", error_id);
		exit(1);
	}
	
	// Set configuration parameters
	dim3 grid_size=(1); dim3 block_size=(size_squares);
	
	// launch kernel
	kernel<<<grid_size, block_size>>>(d_c, size_squares);
	
	// Wait for device to finish?
	cudaDeviceSynchronize();
	
	// copy data back to host
	error_id = cudaMemcpy(squares, d_c, sizeof(ulong)*size_squares,
		cudaMemcpyDeviceToHost);
	if(error_id != cudaSuccess) {
		printf("cudaMemcpy to host  failed with %d\n", error_id);
		exit(1);
	}
	
	printf("Results:\n");
	for(int x = 0; x < root_max+1; x += 1) {
		printf("%ld ", squares[x]);
	}
	printf("\n");	
	
	// free memory
	error_id = cudaFree(d_c);
	if(error_id != cudaSuccess) {
		printf("cudaFree failed with %d\n", error_id);
		exit(1);
	}	
	
	return 0;
}

