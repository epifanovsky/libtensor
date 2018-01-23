#include <libtensor/gen_block_tensor/impl/gen_bto_set_impl.h>
#include "bto_set_impl.h"

namespace libtensor {


template class gen_bto_set< 1, bto_traits<double>, bto_set<1, double> >;
template class gen_bto_set< 2, bto_traits<double>, bto_set<2, double> >;
template class gen_bto_set< 3, bto_traits<double>, bto_set<3, double> >;
template class gen_bto_set< 4, bto_traits<double>, bto_set<4, double> >;
template class gen_bto_set< 5, bto_traits<double>, bto_set<5, double> >;
template class gen_bto_set< 6, bto_traits<double>, bto_set<6, double> >;
template class gen_bto_set< 7, bto_traits<double>, bto_set<7, double> >;
template class gen_bto_set< 8, bto_traits<double>, bto_set<8, double> >;

template class gen_bto_set< 1, bto_traits<float>, bto_set<1, float> >;
template class gen_bto_set< 2, bto_traits<float>, bto_set<2, float> >;
template class gen_bto_set< 3, bto_traits<float>, bto_set<3, float> >;
template class gen_bto_set< 4, bto_traits<float>, bto_set<4, float> >;
template class gen_bto_set< 5, bto_traits<float>, bto_set<5, float> >;
template class gen_bto_set< 6, bto_traits<float>, bto_set<6, float> >;
template class gen_bto_set< 7, bto_traits<float>, bto_set<7, float> >;
template class gen_bto_set< 8, bto_traits<float>, bto_set<8, float> >;


template class bto_set<1, double>;
template class bto_set<2, double>;
template class bto_set<3, double>;
template class bto_set<4, double>;
template class bto_set<5, double>;
template class bto_set<6, double>;
template class bto_set<7, double>;
template class bto_set<8, double>;

template class bto_set<1, float>;
template class bto_set<2, float>;
template class bto_set<3, float>;
template class bto_set<4, float>;
template class bto_set<5, float>;
template class bto_set<6, float>;
template class bto_set<7, float>;
template class bto_set<8, float>;

} // namespace libtensor
