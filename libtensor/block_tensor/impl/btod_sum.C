#include <libtensor/gen_block_tensor/impl/gen_bto_sum_impl.h>
#include "btod_sum_impl.h"

namespace libtensor {


template class gen_bto_sum< 1, btod_traits >;
template class gen_bto_sum< 2, btod_traits >;
template class gen_bto_sum< 3, btod_traits >;
template class gen_bto_sum< 4, btod_traits >;
template class gen_bto_sum< 5, btod_traits >;
template class gen_bto_sum< 6, btod_traits >;
template class gen_bto_sum< 7, btod_traits >;
template class gen_bto_sum< 8, btod_traits >;

template class btod_sum<1>;
template class btod_sum<2>;
template class btod_sum<3>;
template class btod_sum<4>;
template class btod_sum<5>;
template class btod_sum<6>;
template class btod_sum<7>;
template class btod_sum<8>;


} // namespace libtensor
