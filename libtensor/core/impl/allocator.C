#include "allocator_impl.h"

namespace libtensor {


template class allocator<int>;
template class allocator<double>;

template class std_allocator<int>;
template class std_allocator<double>;


} // namespace libtensor
