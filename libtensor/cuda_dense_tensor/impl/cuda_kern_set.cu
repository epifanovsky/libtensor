#include "cuda_kern_set.h"

namespace libtensor {
namespace cuda {


__global__
void kern_set(double *a, double b) {

    a[threadIdx.x + blockIdx.x * blockDim.x] = b;
}


} // namespace cuda
} // namespace libtensor

