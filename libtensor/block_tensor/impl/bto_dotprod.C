#include <libtensor/gen_block_tensor/impl/gen_bto_dotprod_impl.h>
#include "bto_dotprod_impl.h"

namespace libtensor {


template class gen_bto_dotprod< 1, bto_traits<double>, bto_dotprod<1, double> >;
template class gen_bto_dotprod< 2, bto_traits<double>, bto_dotprod<2, double> >;
template class gen_bto_dotprod< 3, bto_traits<double>, bto_dotprod<3, double> >;
template class gen_bto_dotprod< 4, bto_traits<double>, bto_dotprod<4, double> >;
template class gen_bto_dotprod< 5, bto_traits<double>, bto_dotprod<5, double> >;
template class gen_bto_dotprod< 6, bto_traits<double>, bto_dotprod<6, double> >;
template class gen_bto_dotprod< 7, bto_traits<double>, bto_dotprod<7, double> >;
template class gen_bto_dotprod< 8, bto_traits<double>, bto_dotprod<8, double> >;


template class bto_dotprod<1, double>;
template class bto_dotprod<2, double>;
template class bto_dotprod<3, double>;
template class bto_dotprod<4, double>;
template class bto_dotprod<5, double>;
template class bto_dotprod<6, double>;
template class bto_dotprod<7, double>;
template class bto_dotprod<8, double>;

template class gen_bto_dotprod< 1, bto_traits<float>, bto_dotprod<1, float> >;
template class gen_bto_dotprod< 2, bto_traits<float>, bto_dotprod<2, float> >;
template class gen_bto_dotprod< 3, bto_traits<float>, bto_dotprod<3, float> >;
template class gen_bto_dotprod< 4, bto_traits<float>, bto_dotprod<4, float> >;
template class gen_bto_dotprod< 5, bto_traits<float>, bto_dotprod<5, float> >;
template class gen_bto_dotprod< 6, bto_traits<float>, bto_dotprod<6, float> >;
template class gen_bto_dotprod< 7, bto_traits<float>, bto_dotprod<7, float> >;
template class gen_bto_dotprod< 8, bto_traits<float>, bto_dotprod<8, float> >;


template class bto_dotprod<1, float>;
template class bto_dotprod<2, float>;
template class bto_dotprod<3, float>;
template class bto_dotprod<4, float>;
template class bto_dotprod<5, float>;
template class bto_dotprod<6, float>;
template class bto_dotprod<7, float>;
template class bto_dotprod<8, float>;

} // namespace libtensor
