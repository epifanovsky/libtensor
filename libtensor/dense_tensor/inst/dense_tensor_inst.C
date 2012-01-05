#include <libtensor/core/allocator.h>
#include "../dense_tensor.h"
#include "dense_tensor_impl.h"

namespace libtensor {

template class dense_tensor< 0, double, allocator<double> >;
template class dense_tensor< 1, double, allocator<double> >;
template class dense_tensor< 2, double, allocator<double> >;
template class dense_tensor< 3, double, allocator<double> >;
template class dense_tensor< 4, double, allocator<double> >;
template class dense_tensor< 5, double, allocator<double> >;
template class dense_tensor< 6, double, allocator<double> >;

template class dense_tensor< 0, double, std_allocator<double> >;
template class dense_tensor< 1, double, std_allocator<double> >;
template class dense_tensor< 2, double, std_allocator<double> >;
template class dense_tensor< 3, double, std_allocator<double> >;
template class dense_tensor< 4, double, std_allocator<double> >;
template class dense_tensor< 5, double, std_allocator<double> >;
template class dense_tensor< 6, double, std_allocator<double> >;

} // namespace libtensor
