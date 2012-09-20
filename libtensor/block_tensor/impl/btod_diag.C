#include <libtensor/gen_block_tensor/impl/gen_bto_diag_impl.h>
#include "btod_diag_impl.h"

namespace libtensor {


template class gen_bto_diag< 2, 2, btod_traits, btod_diag<2, 2> >;
template class gen_bto_diag< 3, 2, btod_traits, btod_diag<3, 2> >;
template class gen_bto_diag< 3, 3, btod_traits, btod_diag<3, 3> >;
template class gen_bto_diag< 4, 2, btod_traits, btod_diag<4, 2> >;
template class gen_bto_diag< 4, 3, btod_traits, btod_diag<4, 3> >;
template class gen_bto_diag< 4, 4, btod_traits, btod_diag<4, 4> >;
template class gen_bto_diag< 5, 2, btod_traits, btod_diag<5, 2> >;
template class gen_bto_diag< 5, 3, btod_traits, btod_diag<5, 3> >;
template class gen_bto_diag< 5, 4, btod_traits, btod_diag<5, 4> >;
template class gen_bto_diag< 5, 5, btod_traits, btod_diag<5, 5> >;
template class gen_bto_diag< 6, 2, btod_traits, btod_diag<6, 2> >;
template class gen_bto_diag< 6, 3, btod_traits, btod_diag<6, 3> >;
template class gen_bto_diag< 6, 4, btod_traits, btod_diag<6, 4> >;
template class gen_bto_diag< 6, 5, btod_traits, btod_diag<6, 5> >;
template class gen_bto_diag< 6, 6, btod_traits, btod_diag<6, 6> >;

template class btod_diag<2, 2>;
template class btod_diag<3, 2>;
template class btod_diag<3, 3>;
template class btod_diag<4, 2>;
template class btod_diag<4, 3>;
template class btod_diag<4, 4>;
template class btod_diag<5, 2>;
template class btod_diag<5, 3>;
template class btod_diag<5, 4>;
template class btod_diag<5, 5>;
template class btod_diag<6, 2>;
template class btod_diag<6, 3>;
template class btod_diag<6, 4>;
template class btod_diag<6, 5>;
template class btod_diag<6, 6>;


} // namespace libtensor
