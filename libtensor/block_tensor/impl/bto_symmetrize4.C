#include <libtensor/gen_block_tensor/impl/gen_bto_symmetrize4_impl.h>
#include "bto_symmetrize4_impl.h"

namespace libtensor {


template class gen_bto_symmetrize4< 4, bto_traits<double>, bto_symmetrize4<4, double> >;
template class gen_bto_symmetrize4< 5, bto_traits<double>, bto_symmetrize4<5, double> >;
template class gen_bto_symmetrize4< 6, bto_traits<double>, bto_symmetrize4<6, double> >;
template class gen_bto_symmetrize4< 7, bto_traits<double>, bto_symmetrize4<7, double> >;
template class gen_bto_symmetrize4< 8, bto_traits<double>, bto_symmetrize4<8, double> >;

template class bto_symmetrize4<4, double>;
template class bto_symmetrize4<5, double>;
template class bto_symmetrize4<6, double>;
template class bto_symmetrize4<7, double>;
template class bto_symmetrize4<8, double>;

template class gen_bto_symmetrize4< 4, bto_traits<float>, bto_symmetrize4<4, float> >;
template class gen_bto_symmetrize4< 5, bto_traits<float>, bto_symmetrize4<5, float> >;
template class gen_bto_symmetrize4< 6, bto_traits<float>, bto_symmetrize4<6, float> >;
template class gen_bto_symmetrize4< 7, bto_traits<float>, bto_symmetrize4<7, float> >;
template class gen_bto_symmetrize4< 8, bto_traits<float>, bto_symmetrize4<8, float> >;

template class bto_symmetrize4<4, float>;
template class bto_symmetrize4<5, float>;
template class bto_symmetrize4<6, float>;
template class bto_symmetrize4<7, float>;
template class bto_symmetrize4<8, float>;

} // namespace libtensor

