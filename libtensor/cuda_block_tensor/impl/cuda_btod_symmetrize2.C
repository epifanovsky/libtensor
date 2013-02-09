#include <libtensor/gen_block_tensor/impl/gen_bto_symmetrize2_impl.h>
#include "cuda_btod_symmetrize2_impl.h"

namespace libtensor {


template class gen_bto_symmetrize2< 2, cuda_btod_traits,
    cuda_btod_symmetrize2<2> >;
template class gen_bto_symmetrize2< 3, cuda_btod_traits,
    cuda_btod_symmetrize2<3> >;
template class gen_bto_symmetrize2< 4, cuda_btod_traits,
    cuda_btod_symmetrize2<4> >;
template class gen_bto_symmetrize2< 5, cuda_btod_traits,
    cuda_btod_symmetrize2<5> >;
template class gen_bto_symmetrize2< 6, cuda_btod_traits,
    cuda_btod_symmetrize2<6> >;

template class cuda_btod_symmetrize2<2>;
template class cuda_btod_symmetrize2<3>;
template class cuda_btod_symmetrize2<4>;
template class cuda_btod_symmetrize2<5>;
template class cuda_btod_symmetrize2<6>;


} // namespace libtensor

