#include <libtensor/gen_block_tensor/impl/gen_bto_aux_copy_impl.h>
#include <libtensor/diag_block_tensor/diag_btod_traits.h>

namespace libtensor {


template class gen_bto_aux_copy<1, diag_btod_traits>;
template class gen_bto_aux_copy<2, diag_btod_traits>;
template class gen_bto_aux_copy<3, diag_btod_traits>;
template class gen_bto_aux_copy<4, diag_btod_traits>;
template class gen_bto_aux_copy<5, diag_btod_traits>;
template class gen_bto_aux_copy<6, diag_btod_traits>;
template class gen_bto_aux_copy<7, diag_btod_traits>;
template class gen_bto_aux_copy<8, diag_btod_traits>;


} // namespace libtensor
