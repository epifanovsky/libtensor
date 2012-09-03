#ifndef LIBTENSOR_BTOD_SET_IMPL_H
#define LIBTENSOR_BTOD_SET_IMPL_H

#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_set_impl.h>
#include "../btod_set.h"

namespace libtensor {


template<size_t N>
const char *btod_set<N>::k_clazz = "btod_set<N>";


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_IMPL_H
