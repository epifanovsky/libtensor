#include <libtensor/gen_block_tensor/impl/gen_bto_unfold_block_list_impl.h>
#include "../bto_traits.h"

namespace libtensor {


template class gen_bto_unfold_block_list< 1, bto_traits<double> >;
template class gen_bto_unfold_block_list< 2, bto_traits<double> >;
template class gen_bto_unfold_block_list< 3, bto_traits<double> >;
template class gen_bto_unfold_block_list< 4, bto_traits<double> >;
template class gen_bto_unfold_block_list< 5, bto_traits<double> >;
template class gen_bto_unfold_block_list< 6, bto_traits<double> >;
template class gen_bto_unfold_block_list< 7, bto_traits<double> >;
template class gen_bto_unfold_block_list< 8, bto_traits<double> >;

template class gen_bto_unfold_block_list< 1, bto_traits<float> >;
template class gen_bto_unfold_block_list< 2, bto_traits<float> >;
template class gen_bto_unfold_block_list< 3, bto_traits<float> >;
template class gen_bto_unfold_block_list< 4, bto_traits<float> >;
template class gen_bto_unfold_block_list< 5, bto_traits<float> >;
template class gen_bto_unfold_block_list< 6, bto_traits<float> >;
template class gen_bto_unfold_block_list< 7, bto_traits<float> >;
template class gen_bto_unfold_block_list< 8, bto_traits<float> >;

} // namespace libtensor
