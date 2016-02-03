#include <libtensor/gen_block_tensor/impl/gen_bto_set_impl.h>
#include "ctf_btod_set_impl.h"

namespace libtensor {


template class gen_bto_set< 1, ctf_btod_traits, ctf_btod_set<1> >;
template class gen_bto_set< 2, ctf_btod_traits, ctf_btod_set<2> >;
template class gen_bto_set< 3, ctf_btod_traits, ctf_btod_set<3> >;
template class gen_bto_set< 4, ctf_btod_traits, ctf_btod_set<4> >;
template class gen_bto_set< 5, ctf_btod_traits, ctf_btod_set<5> >;
template class gen_bto_set< 6, ctf_btod_traits, ctf_btod_set<6> >;
template class gen_bto_set< 7, ctf_btod_traits, ctf_btod_set<7> >;
template class gen_bto_set< 8, ctf_btod_traits, ctf_btod_set<8> >;

template class ctf_btod_set<1>;
template class ctf_btod_set<2>;
template class ctf_btod_set<3>;
template class ctf_btod_set<4>;
template class ctf_btod_set<5>;
template class ctf_btod_set<6>;
template class ctf_btod_set<7>;
template class ctf_btod_set<8>;


} // namespace libtensor
