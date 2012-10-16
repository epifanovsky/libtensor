#include <libtensor/gen_block_tensor/impl/gen_bto_aux_symmetrize_impl.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


template class gen_bto_aux_symmetrize<1, btod_traits>;
template class gen_bto_aux_symmetrize<2, btod_traits>;
template class gen_bto_aux_symmetrize<3, btod_traits>;
template class gen_bto_aux_symmetrize<4, btod_traits>;
template class gen_bto_aux_symmetrize<5, btod_traits>;
template class gen_bto_aux_symmetrize<6, btod_traits>;
template class gen_bto_aux_symmetrize<7, btod_traits>;
template class gen_bto_aux_symmetrize<8, btod_traits>;


} // namespace libtensor
