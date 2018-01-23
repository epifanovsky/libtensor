#include <libtensor/gen_block_tensor/impl/gen_bto_aux_transform_impl.h>
#include <libtensor/block_tensor/bto_traits.h>

namespace libtensor {


template class gen_bto_aux_transform<1, bto_traits<double> >;
template class gen_bto_aux_transform<2, bto_traits<double> >;
template class gen_bto_aux_transform<3, bto_traits<double> >;
template class gen_bto_aux_transform<4, bto_traits<double> >;
template class gen_bto_aux_transform<5, bto_traits<double> >;
template class gen_bto_aux_transform<6, bto_traits<double> >;
template class gen_bto_aux_transform<7, bto_traits<double> >;
template class gen_bto_aux_transform<8, bto_traits<double> >;

template class gen_bto_aux_transform<1, bto_traits<float> >;
template class gen_bto_aux_transform<2, bto_traits<float> >;
template class gen_bto_aux_transform<3, bto_traits<float> >;
template class gen_bto_aux_transform<4, bto_traits<float> >;
template class gen_bto_aux_transform<5, bto_traits<float> >;
template class gen_bto_aux_transform<6, bto_traits<float> >;
template class gen_bto_aux_transform<7, bto_traits<float> >;
template class gen_bto_aux_transform<8, bto_traits<float> >;

} // namespace libtensor
