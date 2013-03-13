#include <libtensor/gen_block_tensor/impl/gen_bto_symmetrize3_impl.h>
#include "btod_symmetrize3_impl.h"

namespace libtensor {


template class gen_bto_symmetrize3< 3, btod_traits, btod_symmetrize3<3> >;
template class gen_bto_symmetrize3< 4, btod_traits, btod_symmetrize3<4> >;
template class gen_bto_symmetrize3< 5, btod_traits, btod_symmetrize3<5> >;
template class gen_bto_symmetrize3< 6, btod_traits, btod_symmetrize3<6> >;

template class btod_symmetrize3<3>;
template class btod_symmetrize3<4>;
template class btod_symmetrize3<5>;
template class btod_symmetrize3<6>;


} // namespace libtensor

