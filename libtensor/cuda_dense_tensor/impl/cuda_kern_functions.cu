#include "cuda_kern_functions.h"
#include <stdio.h>

//#include <cuda_runtime.h>

namespace libtensor {
namespace cuda {

	//Generic copy N-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
//	__global__ void generic_copy_tensor( const double *a, double *b, size_t size ) {
//		int j;
//		//index in the tensor a
//		j = threadIdx.x + blockIdx.x*blockDim.x;
////
//		while (j < size ) {
//			b[j] =a[j];
//			j += blockIdx.x*gridDim.x;
//		}
//	}

	//Copy 2-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor( const double *a, double *b, const uint2 b_incrs, const uint2 dims) {
		int j, k, ind;
		//index along x dimension
		ind = threadIdx.x;

		//index in the tensor a
//		j = threadIdx.x + blockIdx.x*blockDim.x;
		j = threadIdx.x + blockIdx.x*dims.y;
		//new index in the output array b
		k = threadIdx.x*b_incrs.x + blockIdx.x*b_incrs.y;

//		printf("\nk1 = %d, j1 = %d", k, j);
		while (ind < dims.y) {
//			printf("\nk = %d, j = %d, threadIdx = %d, blockIdx = %d", k, j, threadIdx.x, blockIdx.x);
			b[k] = a[j];
			ind += blockDim.x;
			j += blockDim.x;
			k += blockDim.x*b_incrs.x;
		}
	}

	//Copy 1-dimensional tensor a scaled by factor 'multiply' to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor( const double *a, double *b, const uint2 b_incrs, const uint2 dims, const double multiply) {
		int j, k, ind;
		//index along x dimension
		ind = threadIdx.x;
		//index in the tensor a
		j = threadIdx.x + blockIdx.x*dims.y;
		//new index in the output array b
		k = threadIdx.x*b_incrs.x + blockIdx.x*b_incrs.y;
		while (ind < dims.y) {
			b[k] = a[j]*multiply;
			ind += blockDim.x;
			j += blockDim.x;
			k += blockDim.x*b_incrs.x;
		}
	}

	//Copy 4-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor( const double *a, double *b, const uint4 b_incrs, const uint4 dims) {

		int j, k;
		//index along x dimension
		int ind_x, ind_y;
		//index in the tensor a
		int block_ind_j =  blockIdx.x*dims.x*dims.y + blockIdx.y*dims.x*dims.y*gridDim.x;
		j = threadIdx.x + threadIdx.y*dims.x + blockIdx.x*dims.x*dims.y	+ block_ind_j;
		//new index in the output array b
		int block_ind_k = blockIdx.x*b_incrs.z + blockIdx.y*b_incrs.w;
		k = threadIdx.x*b_incrs.x + threadIdx.y*b_incrs.y + block_ind_k;

		ind_y = threadIdx.y;
		while (ind_y < dims.y) {
			ind_x = threadIdx.x;
			j = ind_x + ind_y*dims.x + block_ind_j;
			k = ind_x*b_incrs.x + ind_y*b_incrs.y + block_ind_k;
			while (ind_x < dims.x) {
				b[k] = a[j];
				ind_x += blockDim.x;
				j += blockDim.x;
				k += blockDim.x*b_incrs.x;
			} //ind_x
			ind_y += blockDim.y;
		} //ind_y

//		int j, k;
//		//index in the tensor a
//		j = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y + blockIdx.y*blockDim.x*blockDim.y*gridDim.x;
//		//new index in the output array b
//		k = threadIdx.x*b_incrs.x + threadIdx.y*b_incrs.y + blockIdx.x*b_incrs.z + blockIdx.y*b_incrs.w;
//
//		b[k] = a[j];
	}

	//Copy 4-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor( const double *a, double *b, const uint4 b_incrs, const uint4 dims, const double multiply) {
		int j, k;
		//index along x dimension
		int ind_x, ind_y;
		//index in the tensor a
		int block_ind_j =  blockIdx.x*dims.x*dims.y + blockIdx.y*dims.x*dims.y*gridDim.x;
		j = threadIdx.x + threadIdx.y*dims.x + blockIdx.x*dims.x*dims.y	+ block_ind_j;
		//new index in the output array b
		int block_ind_k = blockIdx.x*b_incrs.z + blockIdx.y*b_incrs.w;
		k = threadIdx.x*b_incrs.x + threadIdx.y*b_incrs.y + block_ind_k;

		ind_y = threadIdx.y;
		while (ind_y < dims.y) {
			ind_x = threadIdx.x;
			j = ind_x + ind_y*dims.x + block_ind_j;
			k = ind_x*b_incrs.x + ind_y*b_incrs.y + block_ind_k;
			while (ind_x < dims.x) {
				b[k] = a[j]*multiply;
				ind_x += blockDim.x;
				j += blockDim.x;
				k += blockDim.x*b_incrs.x;
			} //ind_x
			ind_y += blockDim.y;
		} //ind_y
	}

	//Copy 6-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor(const double *a, double *b, const uint3 b_incrs1, const uint3 b_incrs2, const uint3 dims) {
		int j, k;
		//index along x dimension
		int ind_x, ind_y, ind_z;

		//index in the tensor a
//		j = threadIdx.x + threadIdx.y*blockDim.x +  threadIdx.z*blockDim.x*blockDim.y + blockIdx.x*blockDim.x*blockDim.y*blockDim.z
//				+ blockIdx.y*blockDim.x*blockDim.y*blockDim.z*gridDim.x + blockIdx.z*blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y;

		int block_ind_j =  blockIdx.x*dims.x*dims.y*dims.z + blockIdx.y*dims.x*dims.y*dims.z*gridDim.x + blockIdx.z*dims.x*dims.y*dims.z*gridDim.x*gridDim.y;
		j = threadIdx.x + threadIdx.y*dims.x +  threadIdx.z*dims.x*dims.y + blockIdx.x*dims.x*dims.y*dims.z
				+ block_ind_j;
//					+ blockIdx.y*dims.x*dims.y*dims.z*gridDim.x + blockIdx.z*dims.x*dims.y*dims.z*gridDim.x*gridDim.y;
		//new index in the output array b
		int block_ind_k = blockIdx.x*b_incrs2.x + blockIdx.y*b_incrs2.y + blockIdx.z*b_incrs2.z;
		k = threadIdx.x*b_incrs1.x + threadIdx.y*b_incrs1.y + threadIdx.z*b_incrs1.z +
				block_ind_k;
//				blockIdx.x*b_incrs2.x + blockIdx.y*b_incrs2.y + + blockIdx.z*b_incrs2.z;

		ind_z = threadIdx.z;
		while ( ind_z < dims.z) {
			ind_y = threadIdx.y;
			while (ind_y < dims.y) {
				ind_x = threadIdx.x;
				j = ind_x + ind_y*dims.x +  ind_z*dims.x*dims.y + block_ind_j;
				k = ind_x*b_incrs1.x + ind_y*b_incrs1.y + ind_z*b_incrs1.z +
								block_ind_k;
				while (ind_x < dims.x) {
//					printf("\nk = %d, j = %d, threadIdx = %d, threadIdy = %d, threadIdz = %d, "
//							"blockIdx = %d, blockIdy = %d, blockIdz = %d",
//							k, j, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
					b[k] = a[j];
					ind_x += blockDim.x;
					j += blockDim.x;
					k += blockDim.x*b_incrs1.x;
				} //ind_x
				ind_y += blockDim.y;
			} //ind_y
			ind_z += blockDim.z;
		} //ind_z
	}

	//Copy 6-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor(const double *a, double *b, const uint3 b_incrs1, const uint3 b_incrs2, const uint3 dims, const double multiply) {
		int j, k;
		//index along x dimension
		int ind_x, ind_y, ind_z;
		//index in the tensor a
		j = threadIdx.x + threadIdx.y*dims.x +  threadIdx.z*dims.x*dims.y + blockIdx.x*dims.x*dims.y*dims.z
					+ blockIdx.y*dims.x*dims.y*dims.z*gridDim.x + blockIdx.z*dims.x*dims.y*dims.z*gridDim.x*gridDim.y;
		int block_ind_j =  blockIdx.x*dims.x*dims.y*dims.z + blockIdx.y*dims.x*dims.y*dims.z*gridDim.x + blockIdx.z*dims.x*dims.y*dims.z*gridDim.x*gridDim.y;
		//new index in the output array b
		k = threadIdx.x*b_incrs1.x + threadIdx.y*b_incrs1.y + threadIdx.z*b_incrs1.z +
				blockIdx.x*b_incrs2.x + blockIdx.y*b_incrs2.y + + blockIdx.z*b_incrs2.z;
		int block_ind_k = blockIdx.x*b_incrs2.x + blockIdx.y*b_incrs2.y + + blockIdx.z*b_incrs2.z;
		ind_z = threadIdx.z;
		while ( ind_z < dims.z) {
			ind_y = threadIdx.y;
			while (ind_y < dims.y) {
				ind_x = threadIdx.x;
				j = ind_x + ind_y*dims.x +  ind_z*dims.x*dims.y + block_ind_j;
				k = ind_x*b_incrs1.x + ind_y*b_incrs1.y + ind_z*b_incrs1.z +
								block_ind_k;
				while (ind_x < dims.x) {
					b[k] = a[j]*multiply;
					ind_x += blockDim.x;
					j += blockDim.x;
					k += blockDim.x*b_incrs1.x;
				} //ind_x
				ind_y += blockDim.y;
			} //ind_y
			ind_z += blockDim.z;
		} //ind_z
	}

	//Add 2-dimensional tensor a multiplied by 'multiply' to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void add_copy_tensor(const double *a, double *b, const uint2 b_incrs, const uint2 dims, const double multiply) {
		int j, k, ind;
		//index along x dimension
		ind = threadIdx.x;
		//index in the tensor a
		j = threadIdx.x + blockIdx.x*dims.y;
		//new index in the output array b
		k = threadIdx.x*b_incrs.x + blockIdx.x*b_incrs.y;

		while (ind < dims.y) {
			b[k] = b[k] + a[j]*multiply;
			ind += blockDim.x;
			j += blockDim.x;
			k += blockDim.x*b_incrs.x;
		}
	}

	//Add 4-dimensional tensor a multiplied by 'multiply' to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void add_copy_tensor(const double *a, double *b, const uint4 b_incrs, const uint4 dims, const double multiply) {
		int j, k;
		//index along x dimension
		int ind_x, ind_y;
		//index in the tensor a
		int block_ind_j =  blockIdx.x*dims.x*dims.y + blockIdx.y*dims.x*dims.y*gridDim.x;
		j = threadIdx.x + threadIdx.y*dims.x + blockIdx.x*dims.x*dims.y	+ block_ind_j;
		//new index in the output array b
		int block_ind_k = blockIdx.x*b_incrs.z + blockIdx.y*b_incrs.w;
		k = threadIdx.x*b_incrs.x + threadIdx.y*b_incrs.y + block_ind_k;

		ind_y = threadIdx.y;
		while (ind_y < dims.y) {
			ind_x = threadIdx.x;
			j = ind_x + ind_y*dims.x + block_ind_j;
			k = ind_x*b_incrs.x + ind_y*b_incrs.y + block_ind_k;
			while (ind_x < dims.x) {
				b[k] = b[k] + a[j]*multiply;
				ind_x += blockDim.x;
				j += blockDim.x;
				k += blockDim.x*b_incrs.x;
			} //ind_x
			ind_y += blockDim.y;
		} //ind_y
	}

	//Add 6-dimensional tensor a multiplied by 'multiply' to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void add_copy_tensor( const double *a, double *b, const uint3 b_incrs1, const uint3 b_incrs2, const uint3 dims, const double multiply) {
		int j, k;
		//index along x dimension
		int ind_x, ind_y, ind_z;
		//index in the tensor a
		j = threadIdx.x + threadIdx.y*dims.x +  threadIdx.z*dims.x*dims.y + blockIdx.x*dims.x*dims.y*dims.z
					+ blockIdx.y*dims.x*dims.y*dims.z*gridDim.x + blockIdx.z*dims.x*dims.y*dims.z*gridDim.x*gridDim.y;
		int block_ind_j =  blockIdx.x*dims.x*dims.y*dims.z + blockIdx.y*dims.x*dims.y*dims.z*gridDim.x + blockIdx.z*dims.x*dims.y*dims.z*gridDim.x*gridDim.y;
		//new index in the output array b
		k = threadIdx.x*b_incrs1.x + threadIdx.y*b_incrs1.y + threadIdx.z*b_incrs1.z +
				blockIdx.x*b_incrs2.x + blockIdx.y*b_incrs2.y + + blockIdx.z*b_incrs2.z;
		int block_ind_k = blockIdx.x*b_incrs2.x + blockIdx.y*b_incrs2.y + + blockIdx.z*b_incrs2.z;
		ind_z = threadIdx.z;
		while ( ind_z < dims.z) {
			ind_y = threadIdx.y;
			while (ind_y < dims.y) {
				ind_x = threadIdx.x;
				j = ind_x + ind_y*dims.x +  ind_z*dims.x*dims.y + block_ind_j;
				k = ind_x*b_incrs1.x + ind_y*b_incrs1.y + ind_z*b_incrs1.z +
								block_ind_k;
				while (ind_x < dims.x) {
					b[k] = b[k] + a[j]*multiply;
					ind_x += blockDim.x;
					j += blockDim.x;
					k += blockDim.x*b_incrs1.x;
				} //ind_x
				ind_y += blockDim.y;
			} //ind_y
			ind_z += blockDim.z;
		} //ind_z

	}



} //namespace cuda
} //namespace libtensor
