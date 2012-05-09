#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/block_tensor/bto/basic_bto.h>
#include <libtensor/block_tensor/bto/impl/basic_bto_impl.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


template class basic_bto<1, bto_traits<double> >;
template class basic_bto<2, bto_traits<double> >;
template class basic_bto<3, bto_traits<double> >;
template class basic_bto<4, bto_traits<double> >;
template class basic_bto<5, bto_traits<double> >;
template class basic_bto<6, bto_traits<double> >;


} // namespace libtensor

