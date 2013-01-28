#include <libtensor/gen_block_tensor/impl/gen_bto_unfold_symmetry_impl.h>
#include "../cuda_btod_traits.h"

namespace libtensor {


template class gen_bto_unfold_symmetry< 1, cuda_btod_traits >;
template class gen_bto_unfold_symmetry< 2, cuda_btod_traits >;
template class gen_bto_unfold_symmetry< 3, cuda_btod_traits >;
template class gen_bto_unfold_symmetry< 4, cuda_btod_traits >;
template class gen_bto_unfold_symmetry< 5, cuda_btod_traits >;
template class gen_bto_unfold_symmetry< 6, cuda_btod_traits >;

} // namespace libtensor
