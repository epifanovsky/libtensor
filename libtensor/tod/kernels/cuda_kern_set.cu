#include "cuda_kern_set.h"
#include <stdio.h>

//#include <cuda_runtime.h>

namespace libtensor {
	namespace cuda {
		/**	\brief Sets all elements of a tensor to the given value
			\tparam a - memory pointer.
			\tparam b - value to be set

			\ingroup libtensor_tod
		 **/
		__global__ void kern_set( double *a, const double b) {
				a[threadIdx.x + blockIdx.x*blockDim.x] = b;
		}
	}
}
