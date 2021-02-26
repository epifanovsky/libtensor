#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_mult1_impl.h>
#include "bto_mult1_impl.h"

namespace libtensor {


template class gen_bto_mult1< 1, bto_traits<double>, bto_mult1<1, double> >;
template class gen_bto_mult1< 2, bto_traits<double>, bto_mult1<2, double> >;
template class gen_bto_mult1< 3, bto_traits<double>, bto_mult1<3, double> >;
template class gen_bto_mult1< 4, bto_traits<double>, bto_mult1<4, double> >;
template class gen_bto_mult1< 5, bto_traits<double>, bto_mult1<5, double> >;
template class gen_bto_mult1< 6, bto_traits<double>, bto_mult1<6, double> >;
template class gen_bto_mult1< 7, bto_traits<double>, bto_mult1<7, double> >;
template class gen_bto_mult1< 8, bto_traits<double>, bto_mult1<8, double> >;


template class bto_mult1<1, double>;
template class bto_mult1<2, double>;
template class bto_mult1<3, double>;
template class bto_mult1<4, double>;
template class bto_mult1<5, double>;
template class bto_mult1<6, double>;
template class bto_mult1<7, double>;
template class bto_mult1<8, double>;

template class gen_bto_mult1< 1, bto_traits<float>, bto_mult1<1, float> >;
template class gen_bto_mult1< 2, bto_traits<float>, bto_mult1<2, float> >;
template class gen_bto_mult1< 3, bto_traits<float>, bto_mult1<3, float> >;
template class gen_bto_mult1< 4, bto_traits<float>, bto_mult1<4, float> >;
template class gen_bto_mult1< 5, bto_traits<float>, bto_mult1<5, float> >;
template class gen_bto_mult1< 6, bto_traits<float>, bto_mult1<6, float> >;
template class gen_bto_mult1< 7, bto_traits<float>, bto_mult1<7, float> >;
template class gen_bto_mult1< 8, bto_traits<float>, bto_mult1<8, float> >;

template class bto_mult1<1, float>;
template class bto_mult1<2, float>;
template class bto_mult1<3, float>;
template class bto_mult1<4, float>;
template class bto_mult1<5, float>;
template class bto_mult1<6, float>;
template class bto_mult1<7, float>;
template class bto_mult1<8, float>;

} // namespace libtensor
