#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_mult_impl.h>
#include "btod_mult_impl.h"

namespace libtensor {


template class gen_bto_mult< 1, btod_traits, btod_mult<1> >;
template class gen_bto_mult< 2, btod_traits, btod_mult<2> >;
template class gen_bto_mult< 3, btod_traits, btod_mult<3> >;
template class gen_bto_mult< 4, btod_traits, btod_mult<4> >;
template class gen_bto_mult< 5, btod_traits, btod_mult<5> >;
template class gen_bto_mult< 6, btod_traits, btod_mult<6> >;
template class gen_bto_mult< 7, btod_traits, btod_mult<7> >;
template class gen_bto_mult< 8, btod_traits, btod_mult<8> >;


template class btod_mult<1>;
template class btod_mult<2>;
template class btod_mult<3>;
template class btod_mult<4>;
template class btod_mult<5>;
template class btod_mult<6>;
template class btod_mult<7>;
template class btod_mult<8>;


} // namespace libtensor
