#ifndef LIBTENSOR_CUDA_KERN_SET_H
#define LIBTENSOR_CUDA_KERN_SET_H

#include <cuda_runtime.h>

namespace libtensor {
namespace cuda {

/**	\brief Sets all elements of a tensor to the given value
    \tparam a - memory pointer.
    \tparam b - value to be set

    \ingroup libtensor_tod
 **/

	__global__ void kern_set( double *a, const double b);



} //namespace cuda
} //namespace libtensor
#endif //LIBTENSOR_CUDA_KERN_SET_H
