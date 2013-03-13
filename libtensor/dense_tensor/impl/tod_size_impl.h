#ifndef LIBTENSOR_TOD_SIZE_IMPL_H
#define LIBTENSOR_TOD_SIZE_IMPL_H

#include <libtensor/core/allocator.h>
#include "../tod_size.h"

namespace libtensor {


template<size_t N>
size_t tod_size<N>::get_size(dense_tensor_rd_i<N, double> &t) {

    size_t n = t.get_dims().get_size();
    return allocator<double>::get_block_size(n);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SIZE_IMPL_H
