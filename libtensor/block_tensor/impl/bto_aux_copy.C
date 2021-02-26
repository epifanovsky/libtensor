#include <libtensor/gen_block_tensor/impl/gen_bto_aux_copy_impl.h>
#include <libtensor/block_tensor/bto_traits.h>

namespace libtensor {


template class gen_bto_aux_copy<1, bto_traits<double> >;
template class gen_bto_aux_copy<2, bto_traits<double> >;
template class gen_bto_aux_copy<3, bto_traits<double> >;
template class gen_bto_aux_copy<4, bto_traits<double> >;
template class gen_bto_aux_copy<5, bto_traits<double> >;
template class gen_bto_aux_copy<6, bto_traits<double> >;
template class gen_bto_aux_copy<7, bto_traits<double> >;
template class gen_bto_aux_copy<8, bto_traits<double> >;

template class gen_bto_aux_copy<1, bto_traits<float> >;
template class gen_bto_aux_copy<2, bto_traits<float> >;
template class gen_bto_aux_copy<3, bto_traits<float> >;
template class gen_bto_aux_copy<4, bto_traits<float> >;
template class gen_bto_aux_copy<5, bto_traits<float> >;
template class gen_bto_aux_copy<6, bto_traits<float> >;
template class gen_bto_aux_copy<7, bto_traits<float> >;
template class gen_bto_aux_copy<8, bto_traits<float> >;

} // namespace libtensor
