#include <libtensor/ctf_dense_tensor/ctf_tod_contract2.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_set.h>
#include "ctf_btod_contract2_impl.h"

namespace libtensor {


template class gen_bto_contract2_simple< 0, 1, 1, ctf_btod_traits,
    ctf_btod_contract2<0, 1, 1> >;
template class gen_bto_contract2_simple< 0, 1, 2, ctf_btod_traits,
    ctf_btod_contract2<0, 1, 2> >;
template class gen_bto_contract2_simple< 0, 1, 3, ctf_btod_traits,
    ctf_btod_contract2<0, 1, 3> >;
template class gen_bto_contract2_simple< 0, 1, 4, ctf_btod_traits,
    ctf_btod_contract2<0, 1, 4> >;
template class gen_bto_contract2_simple< 0, 1, 5, ctf_btod_traits,
    ctf_btod_contract2<0, 1, 5> >;
template class gen_bto_contract2_simple< 0, 1, 6, ctf_btod_traits,
    ctf_btod_contract2<0, 1, 6> >;
template class gen_bto_contract2_simple< 0, 1, 7, ctf_btod_traits,
    ctf_btod_contract2<0, 1, 7> >;
template class gen_bto_contract2_simple< 1, 0, 1, ctf_btod_traits,
    ctf_btod_contract2<1, 0, 1> >;
template class gen_bto_contract2_simple< 1, 0, 2, ctf_btod_traits,
    ctf_btod_contract2<1, 0, 2> >;
template class gen_bto_contract2_simple< 1, 0, 3, ctf_btod_traits,
    ctf_btod_contract2<1, 0, 3> >;
template class gen_bto_contract2_simple< 1, 0, 4, ctf_btod_traits,
    ctf_btod_contract2<1, 0, 4> >;
template class gen_bto_contract2_simple< 1, 0, 5, ctf_btod_traits,
    ctf_btod_contract2<1, 0, 5> >;
template class gen_bto_contract2_simple< 1, 0, 6, ctf_btod_traits,
    ctf_btod_contract2<1, 0, 6> >;
template class gen_bto_contract2_simple< 1, 0, 7, ctf_btod_traits,
    ctf_btod_contract2<1, 0, 7> >;


template class ctf_btod_contract2<0, 1, 1>;
template class ctf_btod_contract2<0, 1, 2>;
template class ctf_btod_contract2<0, 1, 3>;
template class ctf_btod_contract2<0, 1, 4>;
template class ctf_btod_contract2<0, 1, 5>;
template class ctf_btod_contract2<0, 1, 6>;
template class ctf_btod_contract2<0, 1, 7>;
template class ctf_btod_contract2<1, 0, 1>;
template class ctf_btod_contract2<1, 0, 2>;
template class ctf_btod_contract2<1, 0, 3>;
template class ctf_btod_contract2<1, 0, 4>;
template class ctf_btod_contract2<1, 0, 5>;
template class ctf_btod_contract2<1, 0, 6>;
template class ctf_btod_contract2<1, 0, 7>;


} // namespace libtensor
