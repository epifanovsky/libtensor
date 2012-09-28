#include <libtensor/gen_block_tensor/impl/gen_bto_copy_impl.h>
#include "diag_btod_copy_impl.h"

namespace libtensor {


template class gen_bto_copy< 1, diag_btod_traits, diag_btod_copy<1> >;
template class gen_bto_copy< 2, diag_btod_traits, diag_btod_copy<2> >;
template class gen_bto_copy< 3, diag_btod_traits, diag_btod_copy<3> >;
template class gen_bto_copy< 4, diag_btod_traits, diag_btod_copy<4> >;
template class gen_bto_copy< 5, diag_btod_traits, diag_btod_copy<5> >;
template class gen_bto_copy< 6, diag_btod_traits, diag_btod_copy<6> >;

template class diag_btod_copy<1>;
template class diag_btod_copy<2>;
template class diag_btod_copy<3>;
template class diag_btod_copy<4>;
template class diag_btod_copy<5>;
template class diag_btod_copy<6>;


} // namespace libtensor
