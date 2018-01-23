#include <libtensor/gen_block_tensor/impl/gen_bto_size_impl.h>
#include "../bto_traits.h"

namespace libtensor {


template class gen_bto_size<1, bto_traits<double> >;
template class gen_bto_size<2, bto_traits<double> >;
template class gen_bto_size<3, bto_traits<double> >;
template class gen_bto_size<4, bto_traits<double> >;
template class gen_bto_size<5, bto_traits<double> >;
template class gen_bto_size<6, bto_traits<double> >;
template class gen_bto_size<7, bto_traits<double> >;
template class gen_bto_size<8, bto_traits<double> >;

template class gen_bto_size<1, bto_traits<float> >;
template class gen_bto_size<2, bto_traits<float> >;
template class gen_bto_size<3, bto_traits<float> >;
template class gen_bto_size<4, bto_traits<float> >;
template class gen_bto_size<5, bto_traits<float> >;
template class gen_bto_size<6, bto_traits<float> >;
template class gen_bto_size<7, bto_traits<float> >;
template class gen_bto_size<8, bto_traits<float> >;

} // namespace libtensor
