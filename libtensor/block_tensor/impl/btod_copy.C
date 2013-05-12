#include <libtensor/gen_block_tensor/impl/gen_bto_copy_impl.h>
#include "btod_copy_impl.h"

namespace libtensor {


template class gen_bto_copy< 1, btod_traits, btod_copy<1> >;
template class gen_bto_copy< 2, btod_traits, btod_copy<2> >;
template class gen_bto_copy< 3, btod_traits, btod_copy<3> >;
template class gen_bto_copy< 4, btod_traits, btod_copy<4> >;
template class gen_bto_copy< 5, btod_traits, btod_copy<5> >;
template class gen_bto_copy< 6, btod_traits, btod_copy<6> >;
template class gen_bto_copy< 7, btod_traits, btod_copy<7> >;
template class gen_bto_copy< 8, btod_traits, btod_copy<8> >;

template class btod_copy<1>;
template class btod_copy<2>;
template class btod_copy<3>;
template class btod_copy<4>;
template class btod_copy<5>;
template class btod_copy<6>;
template class btod_copy<7>;
template class btod_copy<8>;


} // namespace libtensor
