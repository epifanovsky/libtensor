#include <libtensor/gen_block_tensor/impl/gen_bto_random_impl.h>
#include "bto_random_impl.h"

namespace libtensor {


template class gen_bto_random< 1, bto_traits<double>, bto_random<1, double> >;
template class gen_bto_random< 2, bto_traits<double>, bto_random<2, double> >;
template class gen_bto_random< 3, bto_traits<double>, bto_random<3, double> >;
template class gen_bto_random< 4, bto_traits<double>, bto_random<4, double> >;
template class gen_bto_random< 5, bto_traits<double>, bto_random<5, double> >;
template class gen_bto_random< 6, bto_traits<double>, bto_random<6, double> >;

template class bto_random<1, double>;
template class bto_random<2, double>;
template class bto_random<3, double>;
template class bto_random<4, double>;
template class bto_random<5, double>;
template class bto_random<6, double>;

template class gen_bto_random< 1, bto_traits<float>, bto_random<1, float> >;
template class gen_bto_random< 2, bto_traits<float>, bto_random<2, float> >;
template class gen_bto_random< 3, bto_traits<float>, bto_random<3, float> >;
template class gen_bto_random< 4, bto_traits<float>, bto_random<4, float> >;
template class gen_bto_random< 5, bto_traits<float>, bto_random<5, float> >;
template class gen_bto_random< 6, bto_traits<float>, bto_random<6, float> >;

template class bto_random<1, float>;
template class bto_random<2, float>;
template class bto_random<3, float>;
template class bto_random<4, float>;
template class bto_random<5, float>;
template class bto_random<6, float>;

} // namespace libtensor
