#include <libtensor/gen_block_tensor/impl/gen_bto_symmetrize3_impl.h>
#include "ctf_btod_symmetrize3_impl.h"

namespace libtensor {


template class gen_bto_symmetrize3< 3, ctf_btod_traits, ctf_btod_symmetrize3<3> >;
template class gen_bto_symmetrize3< 4, ctf_btod_traits, ctf_btod_symmetrize3<4> >;
template class gen_bto_symmetrize3< 5, ctf_btod_traits, ctf_btod_symmetrize3<5> >;
template class gen_bto_symmetrize3< 6, ctf_btod_traits, ctf_btod_symmetrize3<6> >;
template class gen_bto_symmetrize3< 7, ctf_btod_traits, ctf_btod_symmetrize3<7> >;
template class gen_bto_symmetrize3< 8, ctf_btod_traits, ctf_btod_symmetrize3<8> >;

template class ctf_btod_symmetrize3<3>;
template class ctf_btod_symmetrize3<4>;
template class ctf_btod_symmetrize3<5>;
template class ctf_btod_symmetrize3<6>;
template class ctf_btod_symmetrize3<7>;
template class ctf_btod_symmetrize3<8>;


} // namespace libtensor

