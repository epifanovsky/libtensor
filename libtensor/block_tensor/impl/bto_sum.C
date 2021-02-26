#include <libtensor/gen_block_tensor/impl/gen_bto_sum_impl.h>
#include "bto_sum_impl.h"

namespace libtensor {


template class gen_bto_sum< 1, bto_traits<double>  >;
template class gen_bto_sum< 2, bto_traits<double>  >;
template class gen_bto_sum< 3, bto_traits<double>  >;
template class gen_bto_sum< 4, bto_traits<double>  >;
template class gen_bto_sum< 5, bto_traits<double>  >;
template class gen_bto_sum< 6, bto_traits<double>  >;
template class gen_bto_sum< 7, bto_traits<double>  >;
template class gen_bto_sum< 8, bto_traits<double>  >;

template class bto_sum<1, double>;
template class bto_sum<2, double>;
template class bto_sum<3, double>;
template class bto_sum<4, double>;
template class bto_sum<5, double>;
template class bto_sum<6, double>;
template class bto_sum<7, double>;
template class bto_sum<8, double>;


template class gen_bto_sum< 1, bto_traits<float>  >;
template class gen_bto_sum< 2, bto_traits<float>  >;
template class gen_bto_sum< 3, bto_traits<float>  >;
template class gen_bto_sum< 4, bto_traits<float>  >;
template class gen_bto_sum< 5, bto_traits<float>  >;
template class gen_bto_sum< 6, bto_traits<float>  >;
template class gen_bto_sum< 7, bto_traits<float>  >;
template class gen_bto_sum< 8, bto_traits<float>  >;

template class bto_sum<1, float>;
template class bto_sum<2, float>;
template class bto_sum<3, float>;
template class bto_sum<4, float>;
template class bto_sum<5, float>;
template class bto_sum<6, float>;
template class bto_sum<7, float>;
template class bto_sum<8, float>;
} // namespace libtensor
