#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_mult1_impl.h>
#include "btod_mult1_impl.h"

namespace libtensor {


template class gen_bto_mult1< 1, btod_traits, btod_mult1<1> >;
template class gen_bto_mult1< 2, btod_traits, btod_mult1<2> >;
template class gen_bto_mult1< 3, btod_traits, btod_mult1<3> >;
template class gen_bto_mult1< 4, btod_traits, btod_mult1<4> >;
template class gen_bto_mult1< 5, btod_traits, btod_mult1<5> >;
template class gen_bto_mult1< 6, btod_traits, btod_mult1<6> >;
template class gen_bto_mult1< 7, btod_traits, btod_mult1<7> >;
template class gen_bto_mult1< 8, btod_traits, btod_mult1<8> >;


template class btod_mult1<1>;
template class btod_mult1<2>;
template class btod_mult1<3>;
template class btod_mult1<4>;
template class btod_mult1<5>;
template class btod_mult1<6>;
template class btod_mult1<7>;
template class btod_mult1<8>;


} // namespace libtensor
