#ifndef LIBTENSOR_TO_SIZE_IMPL_H
#define LIBTENSOR_TO_SIZE_IMPL_H

#include <libtensor/core/allocator.h>
#include "../to_size.h"

namespace libtensor {


template<size_t N, typename T>
size_t to_size<N, T>::get_size(dense_tensor_rd_i<N, T> &t) {

    size_t n = t.get_dims().get_size();
    return allocator::get_block_size(n);
}


} // namespace libtensor

#endif // LIBTENSOR_TO_SIZE_IMPL_H
