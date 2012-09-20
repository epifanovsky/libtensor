#include <libtensor/gen_block_tensor/impl/gen_bto_add_impl.h>
#include "btod_add_impl.h"

namespace libtensor {


template class gen_bto_add< 1, btod_traits, btod_add<1> >;
template class gen_bto_add< 2, btod_traits, btod_add<2> >;
template class gen_bto_add< 3, btod_traits, btod_add<3> >;
template class gen_bto_add< 4, btod_traits, btod_add<4> >;
template class gen_bto_add< 5, btod_traits, btod_add<5> >;
template class gen_bto_add< 6, btod_traits, btod_add<6> >;

template class btod_add<1>;
template class btod_add<2>;
template class btod_add<3>;
template class btod_add<4>;
template class btod_add<5>;
template class btod_add<6>;


} // namespace libtensor
