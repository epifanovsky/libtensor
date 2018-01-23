#include <libtensor/gen_block_tensor/impl/gen_bto_copy_impl.h>
#include "bto_copy_impl.h"

namespace libtensor {


template class gen_bto_copy< 1, bto_traits<double>, bto_copy<1, double> >;
template class gen_bto_copy< 2, bto_traits<double>, bto_copy<2, double> >;
template class gen_bto_copy< 3, bto_traits<double>, bto_copy<3, double> >;
template class gen_bto_copy< 4, bto_traits<double>, bto_copy<4, double> >;
template class gen_bto_copy< 5, bto_traits<double>, bto_copy<5, double> >;
template class gen_bto_copy< 6, bto_traits<double>, bto_copy<6, double> >;
template class gen_bto_copy< 7, bto_traits<double>, bto_copy<7, double> >;
template class gen_bto_copy< 8, bto_traits<double>, bto_copy<8, double> >;

template class bto_copy<1, double>;
template class bto_copy<2, double>;
template class bto_copy<3, double>;
template class bto_copy<4, double>;
template class bto_copy<5, double>;
template class bto_copy<6, double>;
template class bto_copy<7, double>;
template class bto_copy<8, double>;

template class gen_bto_copy< 1, bto_traits<float>, bto_copy<1, float> >;
template class gen_bto_copy< 2, bto_traits<float>, bto_copy<2, float> >;
template class gen_bto_copy< 3, bto_traits<float>, bto_copy<3, float> >;
template class gen_bto_copy< 4, bto_traits<float>, bto_copy<4, float> >;
template class gen_bto_copy< 5, bto_traits<float>, bto_copy<5, float> >;
template class gen_bto_copy< 6, bto_traits<float>, bto_copy<6, float> >;
template class gen_bto_copy< 7, bto_traits<float>, bto_copy<7, float> >;
template class gen_bto_copy< 8, bto_traits<float>, bto_copy<8, float> >;

template class bto_copy<1, float>;
template class bto_copy<2, float>;
template class bto_copy<3, float>;
template class bto_copy<4, float>;
template class bto_copy<5, float>;
template class bto_copy<6, float>;
template class bto_copy<7, float>;
template class bto_copy<8, float>;

} // namespace libtensor
