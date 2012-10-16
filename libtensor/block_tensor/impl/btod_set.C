#include <libtensor/gen_block_tensor/impl/gen_bto_set_impl.h>
#include "btod_set_impl.h"

namespace libtensor {


template class gen_bto_set< 1, btod_traits, btod_set<1> >;
template class gen_bto_set< 2, btod_traits, btod_set<2> >;
template class gen_bto_set< 3, btod_traits, btod_set<3> >;
template class gen_bto_set< 4, btod_traits, btod_set<4> >;
template class gen_bto_set< 5, btod_traits, btod_set<5> >;
template class gen_bto_set< 6, btod_traits, btod_set<6> >;
template class gen_bto_set< 7, btod_traits, btod_set<7> >;
template class gen_bto_set< 8, btod_traits, btod_set<8> >;

template class btod_set<1>;
template class btod_set<2>;
template class btod_set<3>;
template class btod_set<4>;
template class btod_set<5>;
template class btod_set<6>;
template class btod_set<7>;
template class btod_set<8>;


} // namespace libtensor
