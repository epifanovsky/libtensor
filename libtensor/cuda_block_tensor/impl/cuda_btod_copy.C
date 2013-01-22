#include <libtensor/gen_block_tensor/impl/gen_bto_copy_impl.h>
#include "cuda_btod_copy_impl.h"

namespace libtensor {


template class gen_bto_copy< 1, cuda_btod_traits, cuda_btod_copy<1> >;
template class gen_bto_copy< 2, cuda_btod_traits, cuda_btod_copy<2> >;
template class gen_bto_copy< 3, cuda_btod_traits, cuda_btod_copy<3> >;
template class gen_bto_copy< 4, cuda_btod_traits, cuda_btod_copy<4> >;
template class gen_bto_copy< 5, cuda_btod_traits, cuda_btod_copy<5> >;
template class gen_bto_copy< 6, cuda_btod_traits, cuda_btod_copy<6> >;

template class cuda_btod_copy<1>;
template class cuda_btod_copy<2>;
template class cuda_btod_copy<3>;
template class cuda_btod_copy<4>;
template class cuda_btod_copy<5>;
template class cuda_btod_copy<6>;


} // namespace libtensor
