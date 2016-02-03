#include <libtensor/gen_block_tensor/impl/gen_bto_symmetrize2_impl.h>
#include "ctf_btod_symmetrize2_impl.h"

namespace libtensor {


template class gen_bto_symmetrize2< 2, ctf_btod_traits, ctf_btod_symmetrize2<2> >;
template class gen_bto_symmetrize2< 3, ctf_btod_traits, ctf_btod_symmetrize2<3> >;
template class gen_bto_symmetrize2< 4, ctf_btod_traits, ctf_btod_symmetrize2<4> >;
template class gen_bto_symmetrize2< 5, ctf_btod_traits, ctf_btod_symmetrize2<5> >;
template class gen_bto_symmetrize2< 6, ctf_btod_traits, ctf_btod_symmetrize2<6> >;
template class gen_bto_symmetrize2< 7, ctf_btod_traits, ctf_btod_symmetrize2<7> >;
template class gen_bto_symmetrize2< 8, ctf_btod_traits, ctf_btod_symmetrize2<8> >;

template class ctf_btod_symmetrize2<2>;
template class ctf_btod_symmetrize2<3>;
template class ctf_btod_symmetrize2<4>;
template class ctf_btod_symmetrize2<5>;
template class ctf_btod_symmetrize2<6>;
template class ctf_btod_symmetrize2<7>;
template class ctf_btod_symmetrize2<8>;


} // namespace libtensor

