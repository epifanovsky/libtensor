#include <libtensor/gen_block_tensor/impl/gen_bto_add_impl.h>
#include "bto_add_impl.h"

namespace libtensor {


template class gen_bto_add< 1, bto_traits<double>, bto_add<1, double> >;
template class gen_bto_add< 2, bto_traits<double>, bto_add<2, double> >;
template class gen_bto_add< 3, bto_traits<double>, bto_add<3, double> >;
template class gen_bto_add< 4, bto_traits<double>, bto_add<4, double> >;
template class gen_bto_add< 5, bto_traits<double>, bto_add<5, double> >;
template class gen_bto_add< 6, bto_traits<double>, bto_add<6, double> >;

template class bto_add<1, double>;
template class bto_add<2, double>;
template class bto_add<3, double>;
template class bto_add<4, double>;
template class bto_add<5, double>;
template class bto_add<6, double>;

template class gen_bto_add< 1, bto_traits<float>, bto_add<1, float> >;
template class gen_bto_add< 2, bto_traits<float>, bto_add<2, float> >;
template class gen_bto_add< 3, bto_traits<float>, bto_add<3, float> >;
template class gen_bto_add< 4, bto_traits<float>, bto_add<4, float> >;
template class gen_bto_add< 5, bto_traits<float>, bto_add<5, float> >;
template class gen_bto_add< 6, bto_traits<float>, bto_add<6, float> >;

template class bto_add<1, float>;
template class bto_add<2, float>;
template class bto_add<3, float>;
template class bto_add<4, float>;
template class bto_add<5, float>;
template class bto_add<6, float>;

} // namespace libtensor
