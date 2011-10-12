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
	__global__ void copy_tensor( const double *a, double *b, const uint2 b_incrs) {
		int j, k;
		//index in the tensor a
		j = threadIdx.x + blockIdx.x*blockDim.x;
		//new index in the output array b
		k = threadIdx.x*b_incrs.x + blockIdx.x*b_incrs.y;

		b[k] = a[j];
	}

	//Copy 1-dimensional tensor a scaled by factor 'multiply' to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor( const double *a, double *b, const uint2 b_incrs, const double multiply) {
		int j, k;
		//index in the tensor a
		j = threadIdx.x + blockIdx.x*blockDim.x;
		//new index in the output array b
		k = threadIdx.x*b_incrs.x + blockIdx.x*b_incrs.y;
		b[k] = a[j]*multiply;
	}

	//Copy 4-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor( const double *a, double *b, const uint4 b_incrs) {
		int j, k;
		//index in the tensor a
		j = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y + blockIdx.y*blockDim.x*blockDim.y*gridDim.x;
		//new index in the output array b
		k = threadIdx.x*b_incrs.x + threadIdx.y*b_incrs.y + blockIdx.x*b_incrs.z + blockIdx.y*b_incrs.w;

		b[k] = a[j];

//		if (j < 8) {
//		printf("Thread x = %d, y = %d, \n Block x = %d, y = %d \n b[%d] = %f, a[%d] = %f\n",
//				threadIdx.x, threadIdx.y, blockIdx.x,  blockIdx.y, k, b[k], j, a[j]);
//		}
	}

	//Copy 4-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor( const double *a, double *b, const uint4 b_incrs, const double multiply) {
		int j, k;
		//index in the tensor a
		j = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y + blockIdx.y*blockDim.x*blockDim.y*gridDim.x;
		//new index in the output array b
		k = threadIdx.x*b_incrs.x + threadIdx.y*b_incrs.y + blockIdx.x*b_incrs.z + blockIdx.y*b_incrs.w;
		b[k] = a[j]*multiply;
	}

	//Copy 6-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor(const double *a, double *b, const uint3 b_incrs1, const uint3 b_incrs2) {
		int j, k;
		//index in the tensor a
		j = threadIdx.x + threadIdx.y*blockDim.x +  threadIdx.z*blockDim.x*blockDim.y + blockIdx.x*blockDim.x*blockDim.y*blockDim.z
				+ blockIdx.y*blockDim.x*blockDim.y*blockDim.z*gridDim.x + blockIdx.z*blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y;
	//	__shared__ double a_buffer[threadsPerBlock], b_buffer[threadsPerBlock];
		//new index in the output array b
		k = threadIdx.x*b_incrs1.x + threadIdx.y*b_incrs1.y + threadIdx.z*b_incrs1.z + blockIdx.x*b_incrs2.x + blockIdx.y*b_incrs2.y + + blockIdx.z*b_incrs2.z;
		b[k] = a[j];
	}

	//Copy 6-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor(const double *a, double *b, const uint3 b_incrs1, const uint3 b_incrs2, const double multiply) {
		int j, k;
		//index in the tensor a
		j = threadIdx.x + threadIdx.y*blockDim.x +  threadIdx.z*blockDim.x*blockDim.y + blockIdx.x*blockDim.x*blockDim.y*blockDim.z
				+ blockIdx.y*blockDim.x*blockDim.y*blockDim.z*gridDim.x + blockIdx.z*blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y;
	//	__shared__ double a_buffer[threadsPerBlock], b_buffer[threadsPerBlock];
		//new index in the output array b
		k = threadIdx.x*b_incrs1.x + threadIdx.y*b_incrs1.y + threadIdx.z*b_incrs1.z + blockIdx.x*b_incrs2.x + blockIdx.y*b_incrs2.y + + blockIdx.z*b_incrs2.z;
		b[k] = a[j]*multiply;
	}

	//Add 2-dimensional tensor a multiplied by 'multiply' to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void add_copy_tensor(const double *a, double *b, const uint2 b_incrs, const double multiply) {
		int j, k;
		//index in the tensor a
		j = threadIdx.x + blockIdx.x*blockDim.x;
		//new index in the output array b
		k = threadIdx.x*b_incrs.x + blockIdx.x*b_incrs.y;

		b[k] = b[k] + a[j]*multiply;
	}

	//Add 4-dimensional tensor a multiplied by 'multiply' to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void add_copy_tensor(const double *a, double *b, const uint4 b_incrs, const double multiply) {
		int j, k;
		//index in the tensor a
		j = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y + blockIdx.y*blockDim.x*blockDim.y*gridDim.x;
	//	__shared__ double a_buffer[threadsPerBlock], b_buffer[threadsPerBlock];
		//new index in the output array b
		k = threadIdx.x*b_incrs.x + threadIdx.y*b_incrs.y + blockIdx.x*b_incrs.z + blockIdx.y*b_incrs.w;

		b[k] = b[k] + a[j]*multiply;
	}

	//Add 6-dimensional tensor a multiplied by 'multiply' to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void add_copy_tensor( const double *a, double *b, const uint3 b_incrs1, const uint3 b_incrs2, const double multiply) {
			int j, k;
			//index in the tensor a
			j = threadIdx.x + threadIdx.y*blockDim.x +  threadIdx.z*blockDim.x*blockDim.y + blockIdx.x*blockDim.x*blockDim.y*blockDim.z
					+ blockIdx.y*blockDim.x*blockDim.y*blockDim.z*gridDim.x + blockIdx.z*blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y;
		//	__shared__ double a_buffer[threadsPerBlock], b_buffer[threadsPerBlock];
			//new index in the output array b
			k = threadIdx.x*b_incrs1.x + threadIdx.y*b_incrs1.y + threadIdx.z*b_incrs1.z + blockIdx.x*b_incrs2.x + blockIdx.y*b_incrs2.y + + blockIdx.z*b_incrs2.z;

			b[k] = b[k] + a[j]*multiply;
		}



} //namespace cuda
} //namespace libtensor
