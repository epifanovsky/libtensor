#include <libtensor/gen_block_tensor/impl/gen_bto_trace_impl.h>
#include "../bto_trace.h"

namespace libtensor {


template<size_t N, typename T>
const char *bto_trace<N, T>::k_clazz = "bto_trace<N>";

template class gen_bto_trace< 1, bto_traits<double>, bto_trace<1, double> >;
template class gen_bto_trace< 2, bto_traits<double>, bto_trace<2, double> >;
template class gen_bto_trace< 3, bto_traits<double>, bto_trace<3, double> >;
template class gen_bto_trace< 4, bto_traits<double>, bto_trace<4, double> >;

template class bto_trace<1, double>;
template class bto_trace<2, double>;
template class bto_trace<3, double>;
template class bto_trace<4, double>;

template class gen_bto_trace< 1, bto_traits<float>, bto_trace<1, float> >;
template class gen_bto_trace< 2, bto_traits<float>, bto_trace<2, float> >;
template class gen_bto_trace< 3, bto_traits<float>, bto_trace<3, float> >;
template class gen_bto_trace< 4, bto_traits<float>, bto_trace<4, float> >;

template class bto_trace<1, float>;
template class bto_trace<2, float>;
template class bto_trace<3, float>;
template class bto_trace<4, float>;

} // namespace libtensor
