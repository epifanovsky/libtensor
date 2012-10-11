#include <libtensor/gen_block_tensor/impl/gen_bto_random_impl.h>
#include "btod_random_impl.h"

namespace libtensor {


template class gen_bto_random< 1, btod_traits, btod_random<1> >;
template class gen_bto_random< 2, btod_traits, btod_random<2> >;
template class gen_bto_random< 3, btod_traits, btod_random<3> >;
template class gen_bto_random< 4, btod_traits, btod_random<4> >;
template class gen_bto_random< 5, btod_traits, btod_random<5> >;
template class gen_bto_random< 6, btod_traits, btod_random<6> >;

template class btod_random<1>;
template class btod_random<2>;
template class btod_random<3>;
template class btod_random<4>;
template class btod_random<5>;
template class btod_random<6>;


} // namespace libtensor
