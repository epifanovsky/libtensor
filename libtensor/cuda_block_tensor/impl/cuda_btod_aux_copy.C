#include <libtensor/gen_block_tensor/impl/gen_bto_aux_copy_impl.h>
#include <libtensor/cuda_block_tensor/cuda_btod_traits.h>

namespace libtensor {


template class gen_bto_aux_copy<1, cuda_btod_traits>;
template class gen_bto_aux_copy<2, cuda_btod_traits>;
template class gen_bto_aux_copy<3, cuda_btod_traits>;
template class gen_bto_aux_copy<4, cuda_btod_traits>;
template class gen_bto_aux_copy<5, cuda_btod_traits>;
template class gen_bto_aux_copy<6, cuda_btod_traits>;



} // namespace libtensor
