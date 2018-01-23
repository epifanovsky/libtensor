#ifndef LIBTENSOR_BTO_SCALE_IMPL_H
#define LIBTENSOR_BTO_SCALE_IMPL_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_scale_impl.h>
#include "../bto_scale.h"

namespace libtensor {


template<size_t N, typename T>
const char bto_scale<N, T>::k_clazz[] = "bto_scale<N>";


} // namespace libtensor

#endif // LIBTENSOR_BTO_SCALE_IMPL_H

