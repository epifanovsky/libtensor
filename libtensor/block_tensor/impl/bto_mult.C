#include <libtensor/gen_block_tensor/impl/gen_bto_mult_impl.h>
#include "bto_mult_impl.h"

namespace libtensor {


template class gen_bto_mult< 1, bto_traits<double>, bto_mult<1, double> >;
template class gen_bto_mult< 2, bto_traits<double>, bto_mult<2, double> >;
template class gen_bto_mult< 3, bto_traits<double>, bto_mult<3, double> >;
template class gen_bto_mult< 4, bto_traits<double>, bto_mult<4, double> >;
template class gen_bto_mult< 5, bto_traits<double>, bto_mult<5, double> >;
template class gen_bto_mult< 6, bto_traits<double>, bto_mult<6, double> >;
template class gen_bto_mult< 7, bto_traits<double>, bto_mult<7, double> >;
template class gen_bto_mult< 8, bto_traits<double>, bto_mult<8, double> >;


template class bto_mult<1, double>;
template class bto_mult<2, double>;
template class bto_mult<3, double>;
template class bto_mult<4, double>;
template class bto_mult<5, double>;
template class bto_mult<6, double>;
template class bto_mult<7, double>;
template class bto_mult<8, double>;

template class gen_bto_mult< 1, bto_traits<float>, bto_mult<1, float> >;
template class gen_bto_mult< 2, bto_traits<float>, bto_mult<2, float> >;
template class gen_bto_mult< 3, bto_traits<float>, bto_mult<3, float> >;
template class gen_bto_mult< 4, bto_traits<float>, bto_mult<4, float> >;
template class gen_bto_mult< 5, bto_traits<float>, bto_mult<5, float> >;
template class gen_bto_mult< 6, bto_traits<float>, bto_mult<6, float> >;
template class gen_bto_mult< 7, bto_traits<float>, bto_mult<7, float> >;
template class gen_bto_mult< 8, bto_traits<float>, bto_mult<8, float> >;


template class bto_mult<1, float>;
template class bto_mult<2, float>;
template class bto_mult<3, float>;
template class bto_mult<4, float>;
template class bto_mult<5, float>;
template class bto_mult<6, float>;
template class bto_mult<7, float>;
template class bto_mult<8, float>;

} // namespace libtensor
