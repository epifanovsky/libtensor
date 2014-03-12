#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_aux_copy_impl.h>
#include <libtensor/ctf_block_tensor/ctf_btod_traits.h>

namespace libtensor {


template class gen_bto_aux_copy<1, ctf_btod_traits>;
template class gen_bto_aux_copy<2, ctf_btod_traits>;
template class gen_bto_aux_copy<3, ctf_btod_traits>;
template class gen_bto_aux_copy<4, ctf_btod_traits>;
template class gen_bto_aux_copy<5, ctf_btod_traits>;
template class gen_bto_aux_copy<6, ctf_btod_traits>;
template class gen_bto_aux_copy<7, ctf_btod_traits>;
template class gen_bto_aux_copy<8, ctf_btod_traits>;


} // namespace libtensor
