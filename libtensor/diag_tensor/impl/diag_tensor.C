#include <libtensor/core/allocator.h>
#include "diag_tensor_impl.h"

namespace libtensor {


template class diag_tensor<1, double, allocator<double> >;
template class diag_tensor<2, double, allocator<double> >;
template class diag_tensor<3, double, allocator<double> >;
template class diag_tensor<4, double, allocator<double> >;
template class diag_tensor<5, double, allocator<double> >;
template class diag_tensor<6, double, allocator<double> >;
template class diag_tensor<7, double, allocator<double> >;
template class diag_tensor<8, double, allocator<double> >;

template class diag_tensor<1, double, std_allocator<double> >;
template class diag_tensor<2, double, std_allocator<double> >;
template class diag_tensor<3, double, std_allocator<double> >;
template class diag_tensor<4, double, std_allocator<double> >;
template class diag_tensor<5, double, std_allocator<double> >;
template class diag_tensor<6, double, std_allocator<double> >;
template class diag_tensor<7, double, std_allocator<double> >;
template class diag_tensor<8, double, std_allocator<double> >;


} // namespace libtensor

