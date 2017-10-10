#include <libtensor/gen_block_tensor/impl/gen_bto_copy_impl.h>
#include "btod_copy_xm_impl.h"

namespace libtensor {


template class gen_bto_copy< 1, btod_traits, btod_copy_xm<1> >;
template class gen_bto_copy< 2, btod_traits, btod_copy_xm<2> >;
template class gen_bto_copy< 3, btod_traits, btod_copy_xm<3> >;
template class gen_bto_copy< 4, btod_traits, btod_copy_xm<4> >;
template class gen_bto_copy< 5, btod_traits, btod_copy_xm<5> >;
template class gen_bto_copy< 6, btod_traits, btod_copy_xm<6> >;
template class gen_bto_copy< 7, btod_traits, btod_copy_xm<7> >;
template class gen_bto_copy< 8, btod_traits, btod_copy_xm<8> >;

template class btod_copy_xm<1>;
template class btod_copy_xm<2>;
template class btod_copy_xm<3>;
template class btod_copy_xm<4>;
template class btod_copy_xm<5>;
template class btod_copy_xm<6>;
template class btod_copy_xm<7>;
template class btod_copy_xm<8>;


} // namespace libtensor
