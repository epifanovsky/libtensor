#include <libtensor/gen_block_tensor/impl/gen_bto_scale_impl.h>
#include "btod_scale_impl.h"

namespace libtensor {


template class gen_bto_scale< 1, btod_traits, btod_scale<1> >;
template class gen_bto_scale< 2, btod_traits, btod_scale<2> >;
template class gen_bto_scale< 3, btod_traits, btod_scale<3> >;
template class gen_bto_scale< 4, btod_traits, btod_scale<4> >;
template class gen_bto_scale< 5, btod_traits, btod_scale<5> >;
template class gen_bto_scale< 6, btod_traits, btod_scale<6> >;
template class gen_bto_scale< 7, btod_traits, btod_scale<7> >;
template class gen_bto_scale< 8, btod_traits, btod_scale<8> >;

template class btod_scale<1>;
template class btod_scale<2>;
template class btod_scale<3>;
template class btod_scale<4>;
template class btod_scale<5>;
template class btod_scale<6>;
template class btod_scale<7>;
template class btod_scale<8>;


} // namespace libtensor
