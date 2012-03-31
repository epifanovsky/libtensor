#include <libtensor/block_tensor/bto/impl/bto_vmpriority_impl.h>
#include <libtensor/btod/btod_vmpriority.h>

namespace libtensor {


template class bto_vmpriority<1, bto_traits<double> >;
template class bto_vmpriority<2, bto_traits<double> >;
template class bto_vmpriority<3, bto_traits<double> >;
template class bto_vmpriority<4, bto_traits<double> >;
template class bto_vmpriority<5, bto_traits<double> >;
template class bto_vmpriority<6, bto_traits<double> >;
template class bto_vmpriority<7, bto_traits<double> >;
template class bto_vmpriority<8, bto_traits<double> >;


template class btod_vmpriority<1>;
template class btod_vmpriority<2>;
template class btod_vmpriority<3>;
template class btod_vmpriority<4>;
template class btod_vmpriority<5>;
template class btod_vmpriority<6>;
template class btod_vmpriority<7>;
template class btod_vmpriority<8>;


} // namespace libtensor
