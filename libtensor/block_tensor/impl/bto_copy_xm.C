#include <libtensor/gen_block_tensor/impl/gen_bto_copy_impl.h>
#include "bto_copy_xm_impl.h"

namespace libtensor {



template class gen_bto_copy< 1, bto_traits<double>, bto_copy_xm<1, double> >;
template class gen_bto_copy< 2, bto_traits<double>, bto_copy_xm<2, double> >;
template class gen_bto_copy< 3, bto_traits<double>, bto_copy_xm<3, double> >;
template class gen_bto_copy< 4, bto_traits<double>, bto_copy_xm<4, double> >;
template class gen_bto_copy< 5, bto_traits<double>, bto_copy_xm<5, double> >;
template class gen_bto_copy< 6, bto_traits<double>, bto_copy_xm<6, double> >;
template class gen_bto_copy< 7, bto_traits<double>, bto_copy_xm<7, double> >;
template class gen_bto_copy< 8, bto_traits<double>, bto_copy_xm<8, double> >;



template class gen_bto_copy< 1, bto_traits<float>, bto_copy_xm<1, float> >;
template class gen_bto_copy< 2, bto_traits<float>, bto_copy_xm<2, float> >;
template class gen_bto_copy< 3, bto_traits<float>, bto_copy_xm<3, float> >;
template class gen_bto_copy< 4, bto_traits<float>, bto_copy_xm<4, float> >;
template class gen_bto_copy< 5, bto_traits<float>, bto_copy_xm<5, float> >;
template class gen_bto_copy< 6, bto_traits<float>, bto_copy_xm<6, float> >;
template class gen_bto_copy< 7, bto_traits<float>, bto_copy_xm<7, float> >;
template class gen_bto_copy< 8, bto_traits<float>, bto_copy_xm<8, float> >;

template class bto_copy_xm<1, double>;
template class bto_copy_xm<2, double>;
template class bto_copy_xm<3, double>;
template class bto_copy_xm<4, double>;
template class bto_copy_xm<5, double>;
template class bto_copy_xm<6, double>;
template class bto_copy_xm<7, double>;
template class bto_copy_xm<8, double>;


template class bto_copy_xm<1, float>;
template class bto_copy_xm<2, float>;
template class bto_copy_xm<3, float>;
template class bto_copy_xm<4, float>;
template class bto_copy_xm<5, float>;
template class bto_copy_xm<6, float>;
template class bto_copy_xm<7, float>;
template class bto_copy_xm<8, float>;


} // namespace libtensor
