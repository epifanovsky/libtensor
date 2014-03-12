#include <libtensor/gen_block_tensor/impl/gen_bto_copy_impl.h>
#include "ctf_btod_copy_impl.h"

namespace libtensor {


template class gen_bto_copy< 1, ctf_btod_traits, ctf_btod_copy<1> >;
template class gen_bto_copy< 2, ctf_btod_traits, ctf_btod_copy<2> >;
template class gen_bto_copy< 3, ctf_btod_traits, ctf_btod_copy<3> >;
template class gen_bto_copy< 4, ctf_btod_traits, ctf_btod_copy<4> >;
template class gen_bto_copy< 5, ctf_btod_traits, ctf_btod_copy<5> >;
template class gen_bto_copy< 6, ctf_btod_traits, ctf_btod_copy<6> >;
template class gen_bto_copy< 7, ctf_btod_traits, ctf_btod_copy<7> >;
template class gen_bto_copy< 8, ctf_btod_traits, ctf_btod_copy<8> >;

template class ctf_btod_copy<1>;
template class ctf_btod_copy<2>;
template class ctf_btod_copy<3>;
template class ctf_btod_copy<4>;
template class ctf_btod_copy<5>;
template class ctf_btod_copy<6>;
template class ctf_btod_copy<7>;
template class ctf_btod_copy<8>;


} // namespace libtensor
