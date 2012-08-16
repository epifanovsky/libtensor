#include <libtensor/block_tensor/bto/bto_copy.h>
#include <libtensor/block_tensor/bto/impl/bto_copy_impl.h>
#include "../btod_copy.h"

namespace libtensor {


template class bto_copy<1, btod_copy_traits>;
template class bto_copy<2, btod_copy_traits>;
template class bto_copy<3, btod_copy_traits>;
template class bto_copy<4, btod_copy_traits>;
template class bto_copy<5, btod_copy_traits>;
template class bto_copy<6, btod_copy_traits>;


template class btod_copy<1>;
template class btod_copy<2>;
template class btod_copy<3>;
template class btod_copy<4>;
template class btod_copy<5>;
template class btod_copy<6>;


} // namespace libtensor

