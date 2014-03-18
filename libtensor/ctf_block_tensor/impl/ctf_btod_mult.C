#include <libtensor/gen_block_tensor/impl/gen_bto_mult_impl.h>
#include "ctf_btod_mult_impl.h"

namespace libtensor {


template class gen_bto_mult< 1, ctf_btod_traits, ctf_btod_mult<1> >;
template class gen_bto_mult< 2, ctf_btod_traits, ctf_btod_mult<2> >;
template class gen_bto_mult< 3, ctf_btod_traits, ctf_btod_mult<3> >;
template class gen_bto_mult< 4, ctf_btod_traits, ctf_btod_mult<4> >;
template class gen_bto_mult< 5, ctf_btod_traits, ctf_btod_mult<5> >;
template class gen_bto_mult< 6, ctf_btod_traits, ctf_btod_mult<6> >;
template class gen_bto_mult< 7, ctf_btod_traits, ctf_btod_mult<7> >;
template class gen_bto_mult< 8, ctf_btod_traits, ctf_btod_mult<8> >;


template class ctf_btod_mult<1>;
template class ctf_btod_mult<2>;
template class ctf_btod_mult<3>;
template class ctf_btod_mult<4>;
template class ctf_btod_mult<5>;
template class ctf_btod_mult<6>;
template class ctf_btod_mult<7>;
template class ctf_btod_mult<8>;


} // namespace libtensor
