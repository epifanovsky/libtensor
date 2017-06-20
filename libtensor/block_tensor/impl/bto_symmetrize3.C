#include <libtensor/gen_block_tensor/impl/gen_bto_symmetrize3_impl.h>
#include "bto_symmetrize3_impl.h"

namespace libtensor {


template class gen_bto_symmetrize3< 3, bto_traits<double>, bto_symmetrize3<3, double> >;
template class gen_bto_symmetrize3< 4, bto_traits<double>, bto_symmetrize3<4, double> >;
template class gen_bto_symmetrize3< 5, bto_traits<double>, bto_symmetrize3<5, double> >;
template class gen_bto_symmetrize3< 6, bto_traits<double>, bto_symmetrize3<6, double> >;
template class gen_bto_symmetrize3< 7, bto_traits<double>, bto_symmetrize3<7, double> >;
template class gen_bto_symmetrize3< 8, bto_traits<double>, bto_symmetrize3<8, double> >;

template class bto_symmetrize3<3, double>;
template class bto_symmetrize3<4, double>;
template class bto_symmetrize3<5, double>;
template class bto_symmetrize3<6, double>;
template class bto_symmetrize3<7, double>;
template class bto_symmetrize3<8, double>;

template class gen_bto_symmetrize3< 3, bto_traits<float>, bto_symmetrize3<3, float> >;
template class gen_bto_symmetrize3< 4, bto_traits<float>, bto_symmetrize3<4, float> >;
template class gen_bto_symmetrize3< 5, bto_traits<float>, bto_symmetrize3<5, float> >;
template class gen_bto_symmetrize3< 6, bto_traits<float>, bto_symmetrize3<6, float> >;
template class gen_bto_symmetrize3< 7, bto_traits<float>, bto_symmetrize3<7, float> >;
template class gen_bto_symmetrize3< 8, bto_traits<float>, bto_symmetrize3<8, float> >;

template class bto_symmetrize3<3, float>;
template class bto_symmetrize3<4, float>;
template class bto_symmetrize3<5, float>;
template class bto_symmetrize3<6, float>;
template class bto_symmetrize3<7, float>;
template class bto_symmetrize3<8, float>;

} // namespace libtensor

