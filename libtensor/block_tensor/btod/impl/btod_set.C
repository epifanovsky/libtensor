#include <libtensor/block_tensor/bto/bto_set.h>
#include <libtensor/block_tensor/bto/impl/bto_set_impl.h>
#include "../btod_set.h"

namespace libtensor {


template class bto_set<1, btod_set_traits>;
template class bto_set<2, btod_set_traits>;
template class bto_set<3, btod_set_traits>;
template class bto_set<4, btod_set_traits>;
template class bto_set<5, btod_set_traits>;
template class bto_set<6, btod_set_traits>;
template class bto_set<7, btod_set_traits>;
template class bto_set<8, btod_set_traits>;


template class btod_set<1>;
template class btod_set<2>;
template class btod_set<3>;
template class btod_set<4>;
template class btod_set<5>;
template class btod_set<6>;
template class btod_set<7>;
template class btod_set<8>;


} // namespace libtensor
