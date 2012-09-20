#include <libtensor/block_tensor/bto/impl/bto_dotprod_impl.h>
#include "../btod_dotprod.h"

namespace libtensor {


template class bto_dotprod<1, btod_traits>;
template class bto_dotprod<2, btod_traits>;
template class bto_dotprod<3, btod_traits>;
template class bto_dotprod<4, btod_traits>;
template class bto_dotprod<5, btod_traits>;
template class bto_dotprod<6, btod_traits>;

template class btod_dotprod<1>;
template class btod_dotprod<2>;
template class btod_dotprod<3>;
template class btod_dotprod<4>;
template class btod_dotprod<5>;
template class btod_dotprod<6>;


} // namespace libtensor
