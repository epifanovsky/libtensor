#include <libtensor/gen_block_tensor/impl/gen_bto_add_impl.h>
#include "cuda_btod_add_impl.h"

namespace libtensor {


template class gen_bto_add< 1, cuda_btod_traits, cuda_btod_add<1> >;
template class gen_bto_add< 2, cuda_btod_traits, cuda_btod_add<2> >;
template class gen_bto_add< 3, cuda_btod_traits, cuda_btod_add<3> >;
template class gen_bto_add< 4, cuda_btod_traits, cuda_btod_add<4> >;
template class gen_bto_add< 5, cuda_btod_traits, cuda_btod_add<5> >;
template class gen_bto_add< 6, cuda_btod_traits, cuda_btod_add<6> >;

template class cuda_btod_add<1>;
template class cuda_btod_add<2>;
template class cuda_btod_add<3>;
template class cuda_btod_add<4>;
template class cuda_btod_add<5>;
template class cuda_btod_add<6>;


} // namespace libtensor
