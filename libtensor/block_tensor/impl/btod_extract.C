#include <libtensor/gen_block_tensor/impl/gen_bto_extract_impl.h>
#include "btod_extract_impl.h"

namespace libtensor {


template class gen_bto_extract< 2, 1, btod_traits, btod_extract<2, 1> >;
template class gen_bto_extract< 3, 1, btod_traits, btod_extract<3, 1> >;
template class gen_bto_extract< 3, 2, btod_traits, btod_extract<3, 2> >;
template class gen_bto_extract< 4, 1, btod_traits, btod_extract<4, 1> >;
template class gen_bto_extract< 4, 2, btod_traits, btod_extract<4, 2> >;
template class gen_bto_extract< 4, 3, btod_traits, btod_extract<4, 3> >;
template class gen_bto_extract< 5, 1, btod_traits, btod_extract<5, 1> >;
template class gen_bto_extract< 5, 2, btod_traits, btod_extract<5, 2> >;
template class gen_bto_extract< 5, 3, btod_traits, btod_extract<5, 3> >;
template class gen_bto_extract< 5, 4, btod_traits, btod_extract<5, 4> >;
template class gen_bto_extract< 6, 1, btod_traits, btod_extract<6, 1> >;
template class gen_bto_extract< 6, 2, btod_traits, btod_extract<6, 2> >;
template class gen_bto_extract< 6, 3, btod_traits, btod_extract<6, 3> >;
template class gen_bto_extract< 6, 4, btod_traits, btod_extract<6, 4> >;
template class gen_bto_extract< 6, 5, btod_traits, btod_extract<6, 5> >;


template class btod_extract<2, 1>;
template class btod_extract<3, 1>;
template class btod_extract<3, 2>;
template class btod_extract<4, 1>;
template class btod_extract<4, 2>;
template class btod_extract<4, 3>;
template class btod_extract<5, 1>;
template class btod_extract<5, 2>;
template class btod_extract<5, 3>;
template class btod_extract<5, 4>;
template class btod_extract<6, 1>;
template class btod_extract<6, 2>;
template class btod_extract<6, 3>;
template class btod_extract<6, 4>;
template class btod_extract<6, 5>;


} // namespace libtensor

