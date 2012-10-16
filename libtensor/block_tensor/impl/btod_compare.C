#include <libtensor/gen_block_tensor/impl/gen_bto_compare_impl.h>
#include "btod_compare_impl.h"

namespace libtensor {


template class gen_bto_compare<1, btod_traits>;
template class gen_bto_compare<2, btod_traits>;
template class gen_bto_compare<3, btod_traits>;
template class gen_bto_compare<4, btod_traits>;
template class gen_bto_compare<5, btod_traits>;
template class gen_bto_compare<6, btod_traits>;

template class btod_compare<1>;
template class btod_compare<2>;
template class btod_compare<3>;
template class btod_compare<4>;
template class btod_compare<5>;
template class btod_compare<6>;


} // namespace libtensor
