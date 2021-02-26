#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_set.h>
#include "bto_contract2_impl.h"
#include "btod_contract2_xm_impl.h"

namespace libtensor {


template class gen_bto_contract2< 0, 1, 1, btod_traits,
    btod_contract2_xm<0, 1, 1> >;
template class gen_bto_contract2< 0, 1, 2, btod_traits,
    btod_contract2_xm<0, 1, 2> >;
template class gen_bto_contract2< 0, 1, 3, btod_traits,
    btod_contract2_xm<0, 1, 3> >;
template class gen_bto_contract2< 0, 1, 4, btod_traits,
    btod_contract2_xm<0, 1, 4> >;
template class gen_bto_contract2< 0, 1, 5, btod_traits,
    btod_contract2_xm<0, 1, 5> >;
template class gen_bto_contract2< 0, 1, 6, btod_traits,
    btod_contract2_xm<0, 1, 6> >;
template class gen_bto_contract2< 0, 1, 7, btod_traits,
    btod_contract2_xm<0, 1, 7> >;
template class gen_bto_contract2< 1, 0, 1, btod_traits,
    btod_contract2_xm<1, 0, 1> >;
template class gen_bto_contract2< 1, 0, 2, btod_traits,
    btod_contract2_xm<1, 0, 2> >;
template class gen_bto_contract2< 1, 0, 3, btod_traits,
    btod_contract2_xm<1, 0, 3> >;
template class gen_bto_contract2< 1, 0, 4, btod_traits,
    btod_contract2_xm<1, 0, 4> >;
template class gen_bto_contract2< 1, 0, 5, btod_traits,
    btod_contract2_xm<1, 0, 5> >;
template class gen_bto_contract2< 1, 0, 6, btod_traits,
    btod_contract2_xm<1, 0, 6> >;
template class gen_bto_contract2< 1, 0, 7, btod_traits,
    btod_contract2_xm<1, 0, 7> >;


template class btod_contract2_xm<0, 1, 1>;
template class btod_contract2_xm<0, 1, 2>;
template class btod_contract2_xm<0, 1, 3>;
template class btod_contract2_xm<0, 1, 4>;
template class btod_contract2_xm<0, 1, 5>;
template class btod_contract2_xm<0, 1, 6>;
template class btod_contract2_xm<0, 1, 7>;
template class btod_contract2_xm<1, 0, 1>;
template class btod_contract2_xm<1, 0, 2>;
template class btod_contract2_xm<1, 0, 3>;
template class btod_contract2_xm<1, 0, 4>;
template class btod_contract2_xm<1, 0, 5>;
template class btod_contract2_xm<1, 0, 6>;
template class btod_contract2_xm<1, 0, 7>;


} // namespace libtensor
