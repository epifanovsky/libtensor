#include <libtensor/gen_block_tensor/impl/gen_bto_mult1_impl.h>
#include "ctf_btod_mult1_impl.h"

namespace libtensor {


template class gen_bto_mult1< 1, ctf_btod_traits, ctf_btod_mult1<1> >;
template class gen_bto_mult1< 2, ctf_btod_traits, ctf_btod_mult1<2> >;
template class gen_bto_mult1< 3, ctf_btod_traits, ctf_btod_mult1<3> >;
template class gen_bto_mult1< 4, ctf_btod_traits, ctf_btod_mult1<4> >;
template class gen_bto_mult1< 5, ctf_btod_traits, ctf_btod_mult1<5> >;
template class gen_bto_mult1< 6, ctf_btod_traits, ctf_btod_mult1<6> >;
template class gen_bto_mult1< 7, ctf_btod_traits, ctf_btod_mult1<7> >;
template class gen_bto_mult1< 8, ctf_btod_traits, ctf_btod_mult1<8> >;


template class ctf_btod_mult1<1>;
template class ctf_btod_mult1<2>;
template class ctf_btod_mult1<3>;
template class ctf_btod_mult1<4>;
template class ctf_btod_mult1<5>;
template class ctf_btod_mult1<6>;
template class ctf_btod_mult1<7>;
template class ctf_btod_mult1<8>;


} // namespace libtensor
