#include <libtensor/gen_block_tensor/impl/gen_bto_aux_add_impl.h>
#include <libtensor/block_tensor/bto_traits.h>

namespace libtensor {


template class gen_bto_aux_add<1, bto_traits<double> >;
template class gen_bto_aux_add<2, bto_traits<double> >;
template class gen_bto_aux_add<3, bto_traits<double> >;
template class gen_bto_aux_add<4, bto_traits<double> >;
template class gen_bto_aux_add<5, bto_traits<double> >;
template class gen_bto_aux_add<6, bto_traits<double> >;
template class gen_bto_aux_add<7, bto_traits<double> >;
template class gen_bto_aux_add<8, bto_traits<double> >;

template class gen_bto_aux_add<1, bto_traits<float> >;
template class gen_bto_aux_add<2, bto_traits<float> >;
template class gen_bto_aux_add<3, bto_traits<float> >;
template class gen_bto_aux_add<4, bto_traits<float> >;
template class gen_bto_aux_add<5, bto_traits<float> >;
template class gen_bto_aux_add<6, bto_traits<float> >;
template class gen_bto_aux_add<7, bto_traits<float> >;
template class gen_bto_aux_add<8, bto_traits<float> >;

} // namespace libtensor
