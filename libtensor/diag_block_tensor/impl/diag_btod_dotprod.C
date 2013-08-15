#include <libtensor/gen_block_tensor/impl/gen_bto_dotprod_impl.h>
#include "diag_btod_dotprod_impl.h"

namespace libtensor {


template class gen_bto_dotprod<1, diag_btod_traits, diag_btod_dotprod<1> >;
template class gen_bto_dotprod<2, diag_btod_traits, diag_btod_dotprod<2> >;
template class gen_bto_dotprod<3, diag_btod_traits, diag_btod_dotprod<3> >;
template class gen_bto_dotprod<4, diag_btod_traits, diag_btod_dotprod<4> >;
template class gen_bto_dotprod<5, diag_btod_traits, diag_btod_dotprod<5> >;
template class gen_bto_dotprod<6, diag_btod_traits, diag_btod_dotprod<6> >;


template class diag_btod_dotprod<1>;
template class diag_btod_dotprod<2>;
template class diag_btod_dotprod<3>;
template class diag_btod_dotprod<4>;
template class diag_btod_dotprod<5>;
template class diag_btod_dotprod<6>;


} // namespace libtensor
