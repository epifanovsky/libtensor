#include <libtensor/cuda_dense_tensor/cuda_tod_contract2.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_set.h>
#include "cuda_btod_contract2_impl.h"

namespace libtensor {


template class gen_bto_contract2< 0, 1, 1, cuda_btod_traits,
    cuda_btod_contract2<0, 1, 1> >;
template class gen_bto_contract2< 0, 1, 2, cuda_btod_traits,
    cuda_btod_contract2<0, 1, 2> >;
template class gen_bto_contract2< 0, 1, 3, cuda_btod_traits,
    cuda_btod_contract2<0, 1, 3> >;
template class gen_bto_contract2< 0, 1, 4, cuda_btod_traits,
    cuda_btod_contract2<0, 1, 4> >;
template class gen_bto_contract2< 0, 1, 5, cuda_btod_traits,
    cuda_btod_contract2<0, 1, 5> >;
template class gen_bto_contract2< 1, 0, 1, cuda_btod_traits,
    cuda_btod_contract2<1, 0, 1> >;
template class gen_bto_contract2< 1, 0, 2, cuda_btod_traits,
    cuda_btod_contract2<1, 0, 2> >;
template class gen_bto_contract2< 1, 0, 3, cuda_btod_traits,
    cuda_btod_contract2<1, 0, 3> >;
template class gen_bto_contract2< 1, 0, 4, cuda_btod_traits,
    cuda_btod_contract2<1, 0, 4> >;
template class gen_bto_contract2< 1, 0, 5, cuda_btod_traits,
    cuda_btod_contract2<1, 0, 5> >;


template class cuda_btod_contract2<0, 1, 1>;
template class cuda_btod_contract2<0, 1, 2>;
template class cuda_btod_contract2<0, 1, 3>;
template class cuda_btod_contract2<0, 1, 4>;
template class cuda_btod_contract2<0, 1, 5>;
template class cuda_btod_contract2<1, 0, 1>;
template class cuda_btod_contract2<1, 0, 2>;
template class cuda_btod_contract2<1, 0, 3>;
template class cuda_btod_contract2<1, 0, 4>;
template class cuda_btod_contract2<1, 0, 5>;


} // namespace libtensor
