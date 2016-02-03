#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_unfold_block_list_impl.h>
#include "../ctf_btod_traits.h"

namespace libtensor {


template class gen_bto_unfold_block_list< 1, ctf_btod_traits >;
template class gen_bto_unfold_block_list< 2, ctf_btod_traits >;
template class gen_bto_unfold_block_list< 3, ctf_btod_traits >;
template class gen_bto_unfold_block_list< 4, ctf_btod_traits >;
template class gen_bto_unfold_block_list< 5, ctf_btod_traits >;
template class gen_bto_unfold_block_list< 6, ctf_btod_traits >;
template class gen_bto_unfold_block_list< 7, ctf_btod_traits >;
template class gen_bto_unfold_block_list< 8, ctf_btod_traits >;


} // namespace libtensor
