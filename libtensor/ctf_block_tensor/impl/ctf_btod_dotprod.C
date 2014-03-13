#include <libtensor/gen_block_tensor/impl/gen_bto_dotprod_impl.h>
#include "ctf_btod_dotprod_impl.h"

namespace libtensor {


template class gen_bto_dotprod< 1, ctf_btod_traits, ctf_btod_dotprod<1> >;
template class gen_bto_dotprod< 2, ctf_btod_traits, ctf_btod_dotprod<2> >;
template class gen_bto_dotprod< 3, ctf_btod_traits, ctf_btod_dotprod<3> >;
template class gen_bto_dotprod< 4, ctf_btod_traits, ctf_btod_dotprod<4> >;
template class gen_bto_dotprod< 5, ctf_btod_traits, ctf_btod_dotprod<5> >;
template class gen_bto_dotprod< 6, ctf_btod_traits, ctf_btod_dotprod<6> >;
template class gen_bto_dotprod< 7, ctf_btod_traits, ctf_btod_dotprod<7> >;
template class gen_bto_dotprod< 8, ctf_btod_traits, ctf_btod_dotprod<8> >;


template class ctf_btod_dotprod<1>;
template class ctf_btod_dotprod<2>;
template class ctf_btod_dotprod<3>;
template class ctf_btod_dotprod<4>;
template class ctf_btod_dotprod<5>;
template class ctf_btod_dotprod<6>;
template class ctf_btod_dotprod<7>;
template class ctf_btod_dotprod<8>;


} // namespace libtensor
