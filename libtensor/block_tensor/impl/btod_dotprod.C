#include <libtensor/gen_block_tensor/impl/gen_bto_dotprod_impl.h>
#include "btod_dotprod_impl.h"

namespace libtensor {


template class gen_bto_dotprod<1, btod_traits, btod_dotprod<1> >;
template class gen_bto_dotprod<2, btod_traits, btod_dotprod<1> >;
template class gen_bto_dotprod<3, btod_traits, btod_dotprod<1> >;
template class gen_bto_dotprod<4, btod_traits, btod_dotprod<1> >;
template class gen_bto_dotprod<5, btod_traits, btod_dotprod<1> >;
template class gen_bto_dotprod<6, btod_traits, btod_dotprod<1> >;


template class btod_dotprod<1>;
template class btod_dotprod<2>;
template class btod_dotprod<3>;
template class btod_dotprod<4>;
template class btod_dotprod<5>;
template class btod_dotprod<6>;


} // namespace libtensor
