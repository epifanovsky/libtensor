#include <libtensor/gen_block_tensor/impl/gen_bto_aux_transform_impl.h>
#include <libtensor/cuda_block_tensor/cuda_btod_traits.h>

namespace libtensor {


template class gen_bto_aux_transform<1, cuda_btod_traits>;
template class gen_bto_aux_transform<2, cuda_btod_traits>;
template class gen_bto_aux_transform<3, cuda_btod_traits>;
template class gen_bto_aux_transform<4, cuda_btod_traits>;
template class gen_bto_aux_transform<5, cuda_btod_traits>;
template class gen_bto_aux_transform<6, cuda_btod_traits>;
template class gen_bto_aux_transform<7, cuda_btod_traits>;
template class gen_bto_aux_transform<8, cuda_btod_traits>;


} // namespace libtensor
