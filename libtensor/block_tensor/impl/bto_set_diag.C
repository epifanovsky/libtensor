#include <libtensor/gen_block_tensor/impl/gen_bto_set_diag_impl.h>
#include "bto_set_diag_impl.h"

namespace libtensor {


template class gen_bto_set_diag<1, bto_traits<double> >;
template class gen_bto_set_diag<2, bto_traits<double> >;
template class gen_bto_set_diag<3, bto_traits<double> >;
template class gen_bto_set_diag<4, bto_traits<double> >;
template class gen_bto_set_diag<5, bto_traits<double> >;
template class gen_bto_set_diag<6, bto_traits<double> >;
template class gen_bto_set_diag<7, bto_traits<double> >;
template class gen_bto_set_diag<8, bto_traits<double> >;

template class bto_set_diag<1, double>;
template class bto_set_diag<2, double>;
template class bto_set_diag<3, double>;
template class bto_set_diag<4, double>;
template class bto_set_diag<5, double>;
template class bto_set_diag<6, double>;
template class bto_set_diag<7, double>;
template class bto_set_diag<8, double>;

template class gen_bto_set_diag<1, bto_traits<float> >;
template class gen_bto_set_diag<2, bto_traits<float> >;
template class gen_bto_set_diag<3, bto_traits<float> >;
template class gen_bto_set_diag<4, bto_traits<float> >;
template class gen_bto_set_diag<5, bto_traits<float> >;
template class gen_bto_set_diag<6, bto_traits<float> >;
template class gen_bto_set_diag<7, bto_traits<float> >;
template class gen_bto_set_diag<8, bto_traits<float> >;

template class bto_set_diag<1, float>;
template class bto_set_diag<2, float>;
template class bto_set_diag<3, float>;
template class bto_set_diag<4, float>;
template class bto_set_diag<5, float>;
template class bto_set_diag<6, float>;
template class bto_set_diag<7, float>;
template class bto_set_diag<8, float>;

} // namespace libtensor
