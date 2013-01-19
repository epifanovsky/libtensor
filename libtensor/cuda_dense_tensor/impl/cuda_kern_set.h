#ifndef LIBTENSOR_CUDA_KERN_SET_H
#define LIBTENSOR_CUDA_KERN_SET_H

#include <cuda_runtime.h>

namespace libtensor {
namespace cuda {


/** \brief Sets all elements of a tensor to the given value
    \param a Memory pointer.
    \param b Calue to be set.

    \ingroup libtensor_cuda_tod
 **/
__global__
void kern_set(double *a, double b);


} // namespace cuda
} // namespace libtensor

#endif //LIBTENSOR_CUDA_KERN_SET_H
