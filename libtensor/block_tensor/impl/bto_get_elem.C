#include <libtensor/gen_block_tensor/impl/gen_bto_get_elem_impl.h>
#include "bto_get_elem_impl.h"

namespace libtensor {


template class gen_bto_get_elem<1, bto_traits<double> >;
template class gen_bto_get_elem<2, bto_traits<double> >;
template class gen_bto_get_elem<3, bto_traits<double> >;
template class gen_bto_get_elem<4, bto_traits<double> >;
template class gen_bto_get_elem<5, bto_traits<double> >;
template class gen_bto_get_elem<6, bto_traits<double> >;
template class gen_bto_get_elem<7, bto_traits<double> >;
template class gen_bto_get_elem<8, bto_traits<double> >;

template class bto_get_elem<1, double>;
template class bto_get_elem<2, double>;
template class bto_get_elem<3, double>;
template class bto_get_elem<4, double>;
template class bto_get_elem<5, double>;
template class bto_get_elem<6, double>;
template class bto_get_elem<7, double>;
template class bto_get_elem<8, double>;

template class gen_bto_get_elem<1, bto_traits<float> >;
template class gen_bto_get_elem<2, bto_traits<float> >;
template class gen_bto_get_elem<3, bto_traits<float> >;
template class gen_bto_get_elem<4, bto_traits<float> >;
template class gen_bto_get_elem<5, bto_traits<float> >;
template class gen_bto_get_elem<6, bto_traits<float> >;
template class gen_bto_get_elem<7, bto_traits<float> >;
template class gen_bto_get_elem<8, bto_traits<float> >;

template class bto_get_elem<1, float>;
template class bto_get_elem<2, float>;
template class bto_get_elem<3, float>;
template class bto_get_elem<4, float>;
template class bto_get_elem<5, float>;
template class bto_get_elem<6, float>;
template class bto_get_elem<7, float>;
template class bto_get_elem<8, float>;

} // namespace libtensor
