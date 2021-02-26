#include <libtensor/gen_block_tensor/impl/gen_bto_scale_impl.h>
#include "bto_scale_impl.h"

namespace libtensor {


template class gen_bto_scale< 1, bto_traits<double>, bto_scale<1, double> >;
template class gen_bto_scale< 2, bto_traits<double>, bto_scale<2, double> >;
template class gen_bto_scale< 3, bto_traits<double>, bto_scale<3, double> >;
template class gen_bto_scale< 4, bto_traits<double>, bto_scale<4, double> >;
template class gen_bto_scale< 5, bto_traits<double>, bto_scale<5, double> >;
template class gen_bto_scale< 6, bto_traits<double>, bto_scale<6, double> >;
template class gen_bto_scale< 7, bto_traits<double>, bto_scale<7, double> >;
template class gen_bto_scale< 8, bto_traits<double>, bto_scale<8, double> >;

template class bto_scale<1, double>;
template class bto_scale<2, double>;
template class bto_scale<3, double>;
template class bto_scale<4, double>;
template class bto_scale<5, double>;
template class bto_scale<6, double>;
template class bto_scale<7, double>;
template class bto_scale<8, double>;

template class gen_bto_scale< 1, bto_traits<float>, bto_scale<1, float> >;
template class gen_bto_scale< 2, bto_traits<float>, bto_scale<2, float> >;
template class gen_bto_scale< 3, bto_traits<float>, bto_scale<3, float> >;
template class gen_bto_scale< 4, bto_traits<float>, bto_scale<4, float> >;
template class gen_bto_scale< 5, bto_traits<float>, bto_scale<5, float> >;
template class gen_bto_scale< 6, bto_traits<float>, bto_scale<6, float> >;
template class gen_bto_scale< 7, bto_traits<float>, bto_scale<7, float> >;
template class gen_bto_scale< 8, bto_traits<float>, bto_scale<8, float> >;

template class bto_scale<1, float>;
template class bto_scale<2, float>;
template class bto_scale<3, float>;
template class bto_scale<4, float>;
template class bto_scale<5, float>;
template class bto_scale<6, float>;
template class bto_scale<7, float>;
template class bto_scale<8, float>;

} // namespace libtensor
