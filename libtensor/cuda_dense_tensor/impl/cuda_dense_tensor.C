#include <libtensor/cuda/cuda_allocator.h>
#include <libtensor/dense_tensor/impl/dense_tensor_impl.h>

namespace libtensor {

template class dense_tensor< 1, double, cuda_allocator<double> >;
template class dense_tensor< 2, double, cuda_allocator<double> >;
template class dense_tensor< 3, double, cuda_allocator<double> >;
template class dense_tensor< 4, double, cuda_allocator<double> >;
template class dense_tensor< 5, double, cuda_allocator<double> >;
template class dense_tensor< 6, double, cuda_allocator<double> >;
template class dense_tensor< 7, double, cuda_allocator<double> >;
template class dense_tensor< 8, double, cuda_allocator<double> >;


} // namespace libtensor
