#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/bto/basic_bto.h>
#include <libtensor/block_tensor/bto/impl/basic_bto_impl.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


template class basic_bto<1, btod_traits>;
template class basic_bto<2, btod_traits>;
template class basic_bto<3, btod_traits>;
template class basic_bto<4, btod_traits>;
template class basic_bto<5, btod_traits>;
template class basic_bto<6, btod_traits>;


} // namespace libtensor

