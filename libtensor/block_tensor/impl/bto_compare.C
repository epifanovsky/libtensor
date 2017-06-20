#include <libtensor/gen_block_tensor/impl/gen_bto_compare_impl.h>
#include "bto_compare_impl.h"

namespace libtensor {


template class gen_bto_compare<1, bto_traits<double> >;
template class gen_bto_compare<2, bto_traits<double> >;
template class gen_bto_compare<3, bto_traits<double> >;
template class gen_bto_compare<4, bto_traits<double> >;
template class gen_bto_compare<5, bto_traits<double> >;
template class gen_bto_compare<6, bto_traits<double> >;

template class bto_compare<1, double>;
template class bto_compare<2, double>;
template class bto_compare<3, double>;
template class bto_compare<4, double>;
template class bto_compare<5, double>;
template class bto_compare<6, double>;

template class gen_bto_compare<1, bto_traits<float> >;
template class gen_bto_compare<2, bto_traits<float> >;
template class gen_bto_compare<3, bto_traits<float> >;
template class gen_bto_compare<4, bto_traits<float> >;
template class gen_bto_compare<5, bto_traits<float> >;
template class gen_bto_compare<6, bto_traits<float> >;

template class bto_compare<1, float>;
template class bto_compare<2, float>;
template class bto_compare<3, float>;
template class bto_compare<4, float>;
template class bto_compare<5, float>;
template class bto_compare<6, float>;

} // namespace libtensor
