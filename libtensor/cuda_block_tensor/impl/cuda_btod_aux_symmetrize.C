#include <libtensor/gen_block_tensor/impl/gen_bto_aux_symmetrize_impl.h>
#include <libtensor/cuda_block_tensor/cuda_btod_traits.h>

namespace libtensor {


template class gen_bto_aux_symmetrize<1, cuda_btod_traits>;
template class gen_bto_aux_symmetrize<2, cuda_btod_traits>;
template class gen_bto_aux_symmetrize<3, cuda_btod_traits>;
template class gen_bto_aux_symmetrize<4, cuda_btod_traits>;
template class gen_bto_aux_symmetrize<5, cuda_btod_traits>;
template class gen_bto_aux_symmetrize<6, cuda_btod_traits>;
template class gen_bto_aux_symmetrize<7, cuda_btod_traits>;
template class gen_bto_aux_symmetrize<8, cuda_btod_traits>;


} // namespace libtensor
