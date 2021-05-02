/*
 * func_g_serial.cxx
 * 
 * Copyright 2021 Mike <mike@pop-os>
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


#include <iostream>
/*
__global__ void func_g(long *managed_sums, long N, long* d_squares, long nSquares) {
	long i = threadIdx.x + (blockDim.x * blockIdx.x);
	if(i <= N) {
		// search for largest square which divides i
		for(int d = nSquares-1; d >= 0; --d) {
			if((i % d_squares[d]) !=0 ) continue;
			managed_sums[i] = d_squares[d];
			break;
		} //for...
	} // if...
}
*/

void g_N(long* h_sums, long* d_squares, long n, const int n_squares) {
	for(int x = n_squares-1; x >= 0; --x) {
		if((n % d_squares[x])==0) {
			h_sums[n-1] = d_squares[x];
			return;
		}
	}
}

int main(int argc, char **argv)
{
	const long N = 100; // N must be < 10000
	const int n_squares = 10;
	long d_squares[n_squares] = {1,4,9,16,25,36,49,64,81,100};
	long h_sums[N];
	
	// set h_sums to known value
	for(int x = 0; x < N; ++x) h_sums[x] = -1;	
	// call function for each n <= N
	for(long n = 1; n <= N; ++n)
		g_N(h_sums, d_squares, n, n_squares);
	// Output the sums array
	long S = 0;
	for(int x = 0; x < N; ++x) {
		S += h_sums[x];
		printf("%ld  ", h_sums[x]);
	}
	
	printf("\nSum = %ld\n", S);
		
	return 0;
}

