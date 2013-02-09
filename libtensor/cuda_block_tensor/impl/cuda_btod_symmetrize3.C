#include <libtensor/gen_block_tensor/impl/gen_bto_symmetrize3_impl.h>
#include "cuda_btod_symmetrize3_impl.h"

namespace libtensor {


template class gen_bto_symmetrize3< 3, cuda_btod_traits,
    cuda_btod_symmetrize3<3> >;
template class gen_bto_symmetrize3< 4, cuda_btod_traits,
    cuda_btod_symmetrize3<4> >;
template class gen_bto_symmetrize3< 5, cuda_btod_traits,
    cuda_btod_symmetrize3<5> >;
template class gen_bto_symmetrize3< 6, cuda_btod_traits,
    cuda_btod_symmetrize3<6> >;

template class cuda_btod_symmetrize3<3>;
template class cuda_btod_symmetrize3<4>;
template class cuda_btod_symmetrize3<5>;
template class cuda_btod_symmetrize3<6>;


} // namespace libtensor

