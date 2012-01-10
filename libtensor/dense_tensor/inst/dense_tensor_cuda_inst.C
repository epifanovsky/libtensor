#include <libvmm/cuda_allocator.cu>
#include "../dense_tensor.h"
#include "dense_tensor_impl.h"

namespace libtensor {
using libvmm::cuda_allocator;

template class dense_tensor< 0, double, cuda_allocator<double> >;
template class dense_tensor< 1, double, cuda_allocator<double> >;
template class dense_tensor< 2, double, cuda_allocator<double> >;
template class dense_tensor< 3, double, cuda_allocator<double> >;
template class dense_tensor< 4, double, cuda_allocator<double> >;
template class dense_tensor< 5, double, cuda_allocator<double> >;
template class dense_tensor< 6, double, cuda_allocator<double> >;

} // namespace libtensor
