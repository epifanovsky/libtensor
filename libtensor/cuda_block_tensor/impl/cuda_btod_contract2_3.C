#include <libtensor/cuda_dense_tensor/cuda_tod_contract2.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_set.h>
#include "cuda_btod_contract2_impl.h"

namespace libtensor {

template class gen_bto_contract2< 0, 3, 1, cuda_btod_traits,
    cuda_btod_contract2<0, 3, 1> >;
template class gen_bto_contract2< 0, 3, 2, cuda_btod_traits,
    cuda_btod_contract2<0, 3, 2> >;
template class gen_bto_contract2< 0, 3, 3, cuda_btod_traits,
    cuda_btod_contract2<0, 3, 3> >;
template class gen_bto_contract2< 1, 2, 0, cuda_btod_traits,
    cuda_btod_contract2<1, 2, 0> >;
template class gen_bto_contract2< 1, 2, 1, cuda_btod_traits,
    cuda_btod_contract2<1, 2, 1> >;
template class gen_bto_contract2< 1, 2, 2, cuda_btod_traits,
    cuda_btod_contract2<1, 2, 2> >;
template class gen_bto_contract2< 1, 2, 3, cuda_btod_traits,
    cuda_btod_contract2<1, 2, 3> >;
template class gen_bto_contract2< 1, 2, 4, cuda_btod_traits,
    cuda_btod_contract2<1, 2, 4> >;
template class gen_bto_contract2< 2, 1, 0, cuda_btod_traits,
    cuda_btod_contract2<2, 1, 0> >;
template class gen_bto_contract2< 2, 1, 1, cuda_btod_traits,
    cuda_btod_contract2<2, 1, 1> >;
template class gen_bto_contract2< 2, 1, 2, cuda_btod_traits,
    cuda_btod_contract2<2, 1, 2> >;
template class gen_bto_contract2< 2, 1, 3, cuda_btod_traits,
    cuda_btod_contract2<2, 1, 3> >;
template class gen_bto_contract2< 2, 1, 4, cuda_btod_traits,
    cuda_btod_contract2<2, 1, 4> >;
template class gen_bto_contract2< 3, 0, 1, cuda_btod_traits,
    cuda_btod_contract2<3, 0, 1> >;
template class gen_bto_contract2< 3, 0, 2, cuda_btod_traits,
    cuda_btod_contract2<3, 0, 2> >;
template class gen_bto_contract2< 3, 0, 3, cuda_btod_traits,
    cuda_btod_contract2<3, 0, 3> >;


template class cuda_btod_contract2<0, 3, 1>;
template class cuda_btod_contract2<0, 3, 2>;
template class cuda_btod_contract2<0, 3, 3>;
template class cuda_btod_contract2<1, 2, 0>;
template class cuda_btod_contract2<1, 2, 1>;
template class cuda_btod_contract2<1, 2, 2>;
template class cuda_btod_contract2<1, 2, 3>;
template class cuda_btod_contract2<1, 2, 4>;
template class cuda_btod_contract2<2, 1, 0>;
template class cuda_btod_contract2<2, 1, 1>;
template class cuda_btod_contract2<2, 1, 2>;
template class cuda_btod_contract2<2, 1, 3>;
template class cuda_btod_contract2<2, 1, 4>;
template class cuda_btod_contract2<3, 0, 1>;
template class cuda_btod_contract2<3, 0, 2>;
template class cuda_btod_contract2<3, 0, 3>;


} // namespace libtensor
