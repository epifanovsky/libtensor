#include <libtensor/gen_block_tensor/impl/gen_bto_symmetrize4_impl.h>
#include "btod_symmetrize4_impl.h"

namespace libtensor {


template class gen_bto_symmetrize4< 4, btod_traits, btod_symmetrize4<4> >;
template class gen_bto_symmetrize4< 5, btod_traits, btod_symmetrize4<5> >;
template class gen_bto_symmetrize4< 6, btod_traits, btod_symmetrize4<6> >;
template class gen_bto_symmetrize4< 7, btod_traits, btod_symmetrize4<7> >;
template class gen_bto_symmetrize4< 8, btod_traits, btod_symmetrize4<8> >;

template class btod_symmetrize4<4>;
template class btod_symmetrize4<5>;
template class btod_symmetrize4<6>;
template class btod_symmetrize4<7>;
template class btod_symmetrize4<8>;


} // namespace libtensor

