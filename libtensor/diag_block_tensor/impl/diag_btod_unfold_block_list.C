#include <libtensor/gen_block_tensor/impl/gen_bto_unfold_block_list_impl.h>
#include "../diag_btod_traits.h"

namespace libtensor {


template class gen_bto_unfold_block_list<1, diag_btod_traits>;
template class gen_bto_unfold_block_list<2, diag_btod_traits>;
template class gen_bto_unfold_block_list<3, diag_btod_traits>;
template class gen_bto_unfold_block_list<4, diag_btod_traits>;
template class gen_bto_unfold_block_list<5, diag_btod_traits>;
template class gen_bto_unfold_block_list<6, diag_btod_traits>;
template class gen_bto_unfold_block_list<8, diag_btod_traits>;


} // namespace libtensor
