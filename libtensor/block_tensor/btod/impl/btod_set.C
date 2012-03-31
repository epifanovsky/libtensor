#include <libtensor/block_tensor/bto/bto_set.h>
#include <libtensor/block_tensor/bto/impl/bto_set_impl.h>
#include <libtensor/btod/btod_set.h>

namespace libtensor {


template class bto_set<1, bto_traits<double> >;
template class bto_set<2, bto_traits<double> >;
template class bto_set<3, bto_traits<double> >;
template class bto_set<4, bto_traits<double> >;
template class bto_set<5, bto_traits<double> >;
template class bto_set<6, bto_traits<double> >;
template class bto_set<7, bto_traits<double> >;
template class bto_set<8, bto_traits<double> >;


template class btod_set<1>;
template class btod_set<2>;
template class btod_set<3>;
template class btod_set<4>;
template class btod_set<5>;
template class btod_set<6>;
template class btod_set<7>;
template class btod_set<8>;


} // namespace libtensor
