#include <libtensor/gen_block_tensor/impl/gen_bto_scale_impl.h>
#include "ctf_btod_scale_impl.h"

namespace libtensor {


template class gen_bto_scale< 1, ctf_btod_traits, ctf_btod_scale<1> >;
template class gen_bto_scale< 2, ctf_btod_traits, ctf_btod_scale<2> >;
template class gen_bto_scale< 3, ctf_btod_traits, ctf_btod_scale<3> >;
template class gen_bto_scale< 4, ctf_btod_traits, ctf_btod_scale<4> >;
template class gen_bto_scale< 5, ctf_btod_traits, ctf_btod_scale<5> >;
template class gen_bto_scale< 6, ctf_btod_traits, ctf_btod_scale<6> >;
template class gen_bto_scale< 7, ctf_btod_traits, ctf_btod_scale<7> >;
template class gen_bto_scale< 8, ctf_btod_traits, ctf_btod_scale<8> >;

template class ctf_btod_scale<1>;
template class ctf_btod_scale<2>;
template class ctf_btod_scale<3>;
template class ctf_btod_scale<4>;
template class ctf_btod_scale<5>;
template class ctf_btod_scale<6>;
template class ctf_btod_scale<7>;
template class ctf_btod_scale<8>;


} // namespace libtensor
