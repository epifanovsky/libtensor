#ifndef LIBTENSOR_CUDA_KERN_FUNCTIONS_H
#define LIBTENSOR_CUDA_KERN_FUNCTIONS_H

#include <cuda_runtime.h>

namespace libtensor {
namespace cuda {

	//Copy 2-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor( const double *a, double *b, const uint2 b_incrs, const uint2 dims);

	__global__ void copy_tensor( const double *a, double *b, const uint2 b_incrs, const uint2 dims, const double multiply);

	//Copy 4-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor( const double *a, double *b, const uint4 b_incrs, const uint4 dims);

	__global__ void copy_tensor( const double *a, double *b, const uint4 b_incrs, const uint4 dims, const double multiply);

	//Copy 6-dimensional tensor a to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void copy_tensor(const double *a, double *b, const uint3 b_incrs1, const uint3 b_incrs2,
			const uint3 dims, const double multiply);

	__global__ void copy_tensor(const double *a, double *b, const uint3 b_incrs1, const uint3 b_incrs2,
			const uint3 dims);

	//Add N-dimensional tensor a multiplied by 'multiply' to tensor b and applies permutation according to increments in b_incrs.
	//Every thread copies one element from tensor a to tensor b.
	//Position of element in a calculated from the thread ID, position in tensor b calculated from thread ID and increments in b
	__global__ void add_copy_tensor(const double *a, double *b, const uint2 b_incrs, const uint2 dims, const double multiply);

	__global__ void add_copy_tensor(const double *a, double *b, const uint4 b_incrs, const uint4 dims, const double multiply);

	__global__ void add_copy_tensor( const double *a, double *b, const uint3 b_incrs1, const uint3 b_incrs2,
			const uint3 dims, const double multiply);

} //namespace cuda
} //namespace libtensor
#endif //LIBTENSOR_CUDA_KERN_FUNCTIONS_H
