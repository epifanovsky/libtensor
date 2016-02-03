#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_aux_add_impl.h>
#include <libtensor/ctf_block_tensor/ctf_btod_traits.h>

namespace libtensor {


template class gen_bto_aux_add<1, ctf_btod_traits>;
template class gen_bto_aux_add<2, ctf_btod_traits>;
template class gen_bto_aux_add<3, ctf_btod_traits>;
template class gen_bto_aux_add<4, ctf_btod_traits>;
template class gen_bto_aux_add<5, ctf_btod_traits>;
template class gen_bto_aux_add<6, ctf_btod_traits>;
template class gen_bto_aux_add<7, ctf_btod_traits>;
template class gen_bto_aux_add<8, ctf_btod_traits>;


} // namespace libtensor
