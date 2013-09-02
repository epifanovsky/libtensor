#include <libtensor/gen_block_tensor/impl/gen_bto_aux_add_impl.h>
#include <libtensor/diag_block_tensor/diag_btod_traits.h>

namespace libtensor {


template class gen_bto_aux_add<1, diag_btod_traits>;
template class gen_bto_aux_add<2, diag_btod_traits>;
template class gen_bto_aux_add<3, diag_btod_traits>;
template class gen_bto_aux_add<4, diag_btod_traits>;
template class gen_bto_aux_add<5, diag_btod_traits>;
template class gen_bto_aux_add<6, diag_btod_traits>;
template class gen_bto_aux_add<7, diag_btod_traits>;
template class gen_bto_aux_add<8, diag_btod_traits>;


} // namespace libtensor
