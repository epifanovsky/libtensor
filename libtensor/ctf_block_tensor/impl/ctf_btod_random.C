#include <libtensor/gen_block_tensor/impl/gen_bto_random_impl.h>
#include "ctf_btod_random_impl.h"

namespace libtensor {


template class gen_bto_random< 1, ctf_btod_traits, ctf_btod_random<1> >;
template class gen_bto_random< 2, ctf_btod_traits, ctf_btod_random<2> >;
template class gen_bto_random< 3, ctf_btod_traits, ctf_btod_random<3> >;
template class gen_bto_random< 4, ctf_btod_traits, ctf_btod_random<4> >;
template class gen_bto_random< 5, ctf_btod_traits, ctf_btod_random<5> >;
template class gen_bto_random< 6, ctf_btod_traits, ctf_btod_random<6> >;

template class ctf_btod_random<1>;
template class ctf_btod_random<2>;
template class ctf_btod_random<3>;
template class ctf_btod_random<4>;
template class ctf_btod_random<5>;
template class ctf_btod_random<6>;


} // namespace libtensor
