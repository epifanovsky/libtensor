#include <libtensor/core/allocator.h>
#include "dense_tensor_impl.h"

namespace libtensor {


template class dense_tensor< 0, double, allocator >;
template class dense_tensor< 1, double, allocator >;
template class dense_tensor< 2, double, allocator >;
template class dense_tensor< 3, double, allocator >;
template class dense_tensor< 4, double, allocator >;
template class dense_tensor< 5, double, allocator >;
template class dense_tensor< 6, double, allocator >;
template class dense_tensor< 7, double, allocator >;
template class dense_tensor< 8, double, allocator >;


template class dense_tensor< 0, float, allocator >;
template class dense_tensor< 1, float, allocator >;
template class dense_tensor< 2, float, allocator >;
template class dense_tensor< 3, float, allocator >;
template class dense_tensor< 4, float, allocator >;
template class dense_tensor< 5, float, allocator >;
template class dense_tensor< 6, float, allocator >;
template class dense_tensor< 7, float, allocator >;
template class dense_tensor< 8, float, allocator >;
} // namespace libtensor
