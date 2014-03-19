#include <libtensor/gen_block_tensor/impl/gen_bto_sum_impl.h>
#include "ctf_btod_sum_impl.h"

namespace libtensor {


template class gen_bto_sum< 1, ctf_btod_traits >;
template class gen_bto_sum< 2, ctf_btod_traits >;
template class gen_bto_sum< 3, ctf_btod_traits >;
template class gen_bto_sum< 4, ctf_btod_traits >;
template class gen_bto_sum< 5, ctf_btod_traits >;
template class gen_bto_sum< 6, ctf_btod_traits >;
template class gen_bto_sum< 7, ctf_btod_traits >;
template class gen_bto_sum< 8, ctf_btod_traits >;

template class ctf_btod_sum<1>;
template class ctf_btod_sum<2>;
template class ctf_btod_sum<3>;
template class ctf_btod_sum<4>;
template class ctf_btod_sum<5>;
template class ctf_btod_sum<6>;
template class ctf_btod_sum<7>;
template class ctf_btod_sum<8>;


} // namespace libtensor
