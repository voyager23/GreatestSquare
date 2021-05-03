/*
 * std_page.cu
 * 
 * examine paging with kernel launch
 */
 


#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define NL printf("\n")

// Standard Page based on a memory allocation of approx 1.6GiB
#define StdPageX 1024
#define StdPageY 200000
#define Elements StdPageX*StdPageY

//----------------------------------------------------------------------
__global__ void set_squares(long *d_squares, long n_squares) {
	long i = threadIdx.x + (blockIdx.x * blockDim.x);
	if(i < n_squares) d_squares[i] = (i+1)*(i+1);
}

//----------------------------------------------------------------------
__global__ void func_g(long *managed_sums, long N, long* d_squares, long nSquares, int pageidx) {

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
int main(int argc, char **argv) {
	
	const long MaxN = 1e14 + 1;
	cudaError_t error_id;
	long *d_squares = NULL;
	
	// extract target N
	long x = 0;
	if(argc == 2) {
		x = atol(argv[1]);
	} else {
		printf("usage: stdp N\n");
		exit(1);
	}	
	const long N = x;
	if(N > MaxN) {
		printf("target: %ld exceeds program limitations %ld\n", N, MaxN);
		exit(2);
	}
	
	// determine the standard page count nPages for N
	const int nPages = (N / (StdPageX * StdPageY)) + 1;
		
	// determine array dimensions for squares
	const long nSquares = (long)(sqrt(N+1));	// defines size of array	

	printf("target: %ld	nSquares: %ld\n", N, nSquares);
	printf("nPages: %d\n", nPages);
	
	// Allocate space on device for squares array
	error_id = cudaMalloc(&d_squares, sizeof(long)*nSquares);
	if(error_id != cudaSuccess) {
		printf("cudaMalloc squares failed with %d\n", error_id);
		exit(1);
	}
	
	// launch the generator on kernel
	printf("Generating squares array...");
	cudaGetLastError(); // set cuda success to 1
	set_squares<<< ((nSquares/1024)+1), 1024 >>>(d_squares, nSquares);
	error_id = cudaPeekAtLastError();
	if(error_id != cudaSuccess) {
		printf("set_squares failed with %s\n", cudaGetErrorString(error_id));
		exit(1);
	}
	printf("done.\n");
	
	// Allocate managed memory for standard page
	cudaDeviceSynchronize();		
	long* managed_sums = NULL;
	error_id = cudaMallocManaged(&managed_sums, sizeof(long)*Elements);
	if(error_id != cudaSuccess) {
		printf("cudaMallocManaged sums failed with %d\n", error_id);
		exit(1);
	}
	printf("Allocated Managed memory: %d blocks of %d threads.\n", StdPageY, StdPageX);
	
	// Now dp nPages of kernel launches
	for(int pageidx = 0; pageidx < nPages; ++pageidx) {
		printf("page:%d\n", pageidx);
		// launch a kernel using calculated configuration
		func_g<<<StdPageY, StdPageX>>>(managed_sums, N, d_squares, nSquares, pageidx);
		cudaDeviceSynchronize();		
	}
	// Clean up code
	NL;
	cudaFree(managed_sums);
	cudaFree(d_squares);
	
}

