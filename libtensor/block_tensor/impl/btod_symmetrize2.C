#include <libtensor/gen_block_tensor/impl/gen_bto_symmetrize2_impl.h>
#include "btod_symmetrize2_impl.h"

namespace libtensor {


template class gen_bto_symmetrize2< 2, btod_traits, btod_symmetrize2<2> >;
template class gen_bto_symmetrize2< 3, btod_traits, btod_symmetrize2<3> >;
template class gen_bto_symmetrize2< 4, btod_traits, btod_symmetrize2<4> >;
template class gen_bto_symmetrize2< 5, btod_traits, btod_symmetrize2<5> >;
template class gen_bto_symmetrize2< 6, btod_traits, btod_symmetrize2<6> >;
template class gen_bto_symmetrize2< 7, btod_traits, btod_symmetrize2<7> >;
template class gen_bto_symmetrize2< 8, btod_traits, btod_symmetrize2<8> >;

template class btod_symmetrize2<2>;
template class btod_symmetrize2<3>;
template class btod_symmetrize2<4>;
template class btod_symmetrize2<5>;
template class btod_symmetrize2<6>;
template class btod_symmetrize2<7>;
template class btod_symmetrize2<8>;


} // namespace libtensor

