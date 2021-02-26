#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_set.h>
#include "bto_contract2_impl.h"
#include "btod_contract2_xm_impl.h"

namespace libtensor {


template class gen_bto_contract2< 0, 5, 1, btod_traits,
    btod_contract2_xm<0, 5, 1> >;
template class gen_bto_contract2< 0, 5, 2, btod_traits,
    btod_contract2_xm<0, 5, 2> >;
template class gen_bto_contract2< 0, 5, 3, btod_traits,
    btod_contract2_xm<0, 5, 3> >;
template class gen_bto_contract2< 1, 4, 0, btod_traits,
    btod_contract2_xm<1, 4, 0> >;
template class gen_bto_contract2< 1, 4, 1, btod_traits,
    btod_contract2_xm<1, 4, 1> >;
template class gen_bto_contract2< 1, 4, 2, btod_traits,
    btod_contract2_xm<1, 4, 2> >;
template class gen_bto_contract2< 1, 4, 3, btod_traits,
    btod_contract2_xm<1, 4, 3> >;
template class gen_bto_contract2< 1, 4, 4, btod_traits,
    btod_contract2_xm<1, 4, 4> >;
template class gen_bto_contract2< 2, 3, 0, btod_traits,
    btod_contract2_xm<2, 3, 0> >;
template class gen_bto_contract2< 2, 3, 1, btod_traits,
    btod_contract2_xm<2, 3, 1> >;
template class gen_bto_contract2< 2, 3, 2, btod_traits,
    btod_contract2_xm<2, 3, 2> >;
template class gen_bto_contract2< 2, 3, 3, btod_traits,
    btod_contract2_xm<2, 3, 3> >;
template class gen_bto_contract2< 2, 3, 4, btod_traits,
    btod_contract2_xm<2, 3, 4> >;
template class gen_bto_contract2< 2, 3, 5, btod_traits,
    btod_contract2_xm<2, 3, 5> >;
template class gen_bto_contract2< 3, 2, 0, btod_traits,
    btod_contract2_xm<3, 2, 0> >;
template class gen_bto_contract2< 3, 2, 1, btod_traits,
    btod_contract2_xm<3, 2, 1> >;
template class gen_bto_contract2< 3, 2, 2, btod_traits,
    btod_contract2_xm<3, 2, 2> >;
template class gen_bto_contract2< 3, 2, 3, btod_traits,
    btod_contract2_xm<3, 2, 3> >;
template class gen_bto_contract2< 3, 2, 4, btod_traits,
    btod_contract2_xm<3, 2, 4> >;
template class gen_bto_contract2< 3, 2, 5, btod_traits,
    btod_contract2_xm<3, 2, 5> >;
template class gen_bto_contract2< 4, 1, 0, btod_traits,
    btod_contract2_xm<4, 1, 0> >;
template class gen_bto_contract2< 4, 1, 1, btod_traits,
    btod_contract2_xm<4, 1, 1> >;
template class gen_bto_contract2< 4, 1, 2, btod_traits,
    btod_contract2_xm<4, 1, 2> >;
template class gen_bto_contract2< 4, 1, 3, btod_traits,
    btod_contract2_xm<4, 1, 3> >;
template class gen_bto_contract2< 4, 1, 4, btod_traits,
    btod_contract2_xm<4, 1, 4> >;
template class gen_bto_contract2< 5, 0, 1, btod_traits,
    btod_contract2_xm<5, 0, 1> >;
template class gen_bto_contract2< 5, 0, 2, btod_traits,
    btod_contract2_xm<5, 0, 2> >;
template class gen_bto_contract2< 5, 0, 3, btod_traits,
    btod_contract2_xm<5, 0, 3> >;


template class btod_contract2_xm<0, 5, 1>;
template class btod_contract2_xm<0, 5, 2>;
template class btod_contract2_xm<0, 5, 3>;
template class btod_contract2_xm<1, 4, 0>;
template class btod_contract2_xm<1, 4, 1>;
template class btod_contract2_xm<1, 4, 2>;
template class btod_contract2_xm<1, 4, 3>;
template class btod_contract2_xm<1, 4, 4>;
template class btod_contract2_xm<2, 3, 0>;
template class btod_contract2_xm<2, 3, 1>;
template class btod_contract2_xm<2, 3, 2>;
template class btod_contract2_xm<2, 3, 3>;
template class btod_contract2_xm<2, 3, 4>;
template class btod_contract2_xm<2, 3, 5>;
template class btod_contract2_xm<3, 2, 0>;
template class btod_contract2_xm<3, 2, 1>;
template class btod_contract2_xm<3, 2, 2>;
template class btod_contract2_xm<3, 2, 3>;
template class btod_contract2_xm<3, 2, 4>;
template class btod_contract2_xm<3, 2, 5>;
template class btod_contract2_xm<4, 1, 0>;
template class btod_contract2_xm<4, 1, 1>;
template class btod_contract2_xm<4, 1, 2>;
template class btod_contract2_xm<4, 1, 3>;
template class btod_contract2_xm<4, 1, 4>;
template class btod_contract2_xm<5, 0, 1>;
template class btod_contract2_xm<5, 0, 2>;
template class btod_contract2_xm<5, 0, 3>;


} // namespace libtensor
