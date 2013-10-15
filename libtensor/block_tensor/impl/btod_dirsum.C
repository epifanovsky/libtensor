#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_dirsum_impl.h>
#include "btod_dirsum_impl.h"

namespace libtensor {


template class gen_bto_dirsum<1, 1, btod_traits, btod_dirsum<1, 1> >;
template class gen_bto_dirsum<1, 2, btod_traits, btod_dirsum<1, 2> >;
template class gen_bto_dirsum<1, 3, btod_traits, btod_dirsum<1, 3> >;
template class gen_bto_dirsum<1, 4, btod_traits, btod_dirsum<1, 4> >;
template class gen_bto_dirsum<1, 5, btod_traits, btod_dirsum<1, 5> >;
template class gen_bto_dirsum<2, 1, btod_traits, btod_dirsum<2, 1> >;
template class gen_bto_dirsum<2, 2, btod_traits, btod_dirsum<2, 2> >;
template class gen_bto_dirsum<2, 3, btod_traits, btod_dirsum<2, 3> >;
template class gen_bto_dirsum<2, 4, btod_traits, btod_dirsum<2, 4> >;
template class gen_bto_dirsum<3, 1, btod_traits, btod_dirsum<3, 1> >;
template class gen_bto_dirsum<3, 2, btod_traits, btod_dirsum<3, 2> >;
template class gen_bto_dirsum<3, 3, btod_traits, btod_dirsum<3, 3> >;
template class gen_bto_dirsum<4, 1, btod_traits, btod_dirsum<4, 1> >;
template class gen_bto_dirsum<4, 2, btod_traits, btod_dirsum<4, 2> >;
template class gen_bto_dirsum<4, 4, btod_traits, btod_dirsum<4, 4> >;
template class gen_bto_dirsum<5, 1, btod_traits, btod_dirsum<5, 1> >;


template class btod_dirsum<1, 1>;
template class btod_dirsum<1, 2>;
template class btod_dirsum<1, 3>;
template class btod_dirsum<1, 4>;
template class btod_dirsum<1, 5>;
template class btod_dirsum<2, 1>;
template class btod_dirsum<2, 2>;
template class btod_dirsum<2, 3>;
template class btod_dirsum<2, 4>;
template class btod_dirsum<3, 1>;
template class btod_dirsum<3, 2>;
template class btod_dirsum<3, 3>;
template class btod_dirsum<4, 1>;
template class btod_dirsum<4, 2>;
template class btod_dirsum<4, 4>;
template class btod_dirsum<5, 1>;


} // namespace libtensor
