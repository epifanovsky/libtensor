#include <libtensor/gen_block_tensor/impl/gen_bto_unfold_block_list_impl.h>
#include "../cuda_btod_traits.h"

namespace libtensor {


template class gen_bto_unfold_block_list< 1, cuda_btod_traits >;
template class gen_bto_unfold_block_list< 2, cuda_btod_traits >;
template class gen_bto_unfold_block_list< 3, cuda_btod_traits >;
template class gen_bto_unfold_block_list< 4, cuda_btod_traits >;
template class gen_bto_unfold_block_list< 5, cuda_btod_traits >;
template class gen_bto_unfold_block_list< 6, cuda_btod_traits >;
template class gen_bto_unfold_block_list< 7, cuda_btod_traits >;
template class gen_bto_unfold_block_list< 8, cuda_btod_traits >;


} // namespace libtensor
