#include <libtensor/gen_block_tensor/impl/gen_bto_aux_add_impl.h>
#include <libtensor/cuda_block_tensor/cuda_btod_traits.h>

namespace libtensor {


template class gen_bto_aux_add<1, cuda_btod_traits>;
template class gen_bto_aux_add<2, cuda_btod_traits>;
template class gen_bto_aux_add<3, cuda_btod_traits>;
template class gen_bto_aux_add<4, cuda_btod_traits>;
template class gen_bto_aux_add<5, cuda_btod_traits>;
template class gen_bto_aux_add<6, cuda_btod_traits>;



} // namespace libtensor
