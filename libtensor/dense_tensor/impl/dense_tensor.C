#include <libtensor/core/allocator.h>
#include "dense_tensor_impl.h"

namespace libtensor {


template class dense_tensor< 0, double, allocator<double> >;
template class dense_tensor< 1, double, allocator<double> >;
template class dense_tensor< 2, double, allocator<double> >;
template class dense_tensor< 3, double, allocator<double> >;
template class dense_tensor< 4, double, allocator<double> >;
template class dense_tensor< 5, double, allocator<double> >;
template class dense_tensor< 6, double, allocator<double> >;
template class dense_tensor< 7, double, allocator<double> >;
template class dense_tensor< 8, double, allocator<double> >;


template class dense_tensor< 0, float, allocator<float> >;
template class dense_tensor< 1, float, allocator<float> >;
template class dense_tensor< 2, float, allocator<float> >;
template class dense_tensor< 3, float, allocator<float> >;
template class dense_tensor< 4, float, allocator<float> >;
template class dense_tensor< 5, float, allocator<float> >;
template class dense_tensor< 6, float, allocator<float> >;
template class dense_tensor< 7, float, allocator<float> >;
template class dense_tensor< 8, float, allocator<float> >;
} // namespace libtensor
