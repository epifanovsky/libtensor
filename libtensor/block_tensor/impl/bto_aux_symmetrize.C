#include <libtensor/gen_block_tensor/impl/gen_bto_aux_symmetrize_impl.h>
#include <libtensor/block_tensor/bto_traits.h>

namespace libtensor {


template class gen_bto_aux_symmetrize<1, bto_traits<double> >;
template class gen_bto_aux_symmetrize<2, bto_traits<double> >;
template class gen_bto_aux_symmetrize<3, bto_traits<double> >;
template class gen_bto_aux_symmetrize<4, bto_traits<double> >;
template class gen_bto_aux_symmetrize<5, bto_traits<double> >;
template class gen_bto_aux_symmetrize<6, bto_traits<double> >;
template class gen_bto_aux_symmetrize<7, bto_traits<double> >;
template class gen_bto_aux_symmetrize<8, bto_traits<double> >;

template class gen_bto_aux_symmetrize<1, bto_traits<float> >;
template class gen_bto_aux_symmetrize<2, bto_traits<float> >;
template class gen_bto_aux_symmetrize<3, bto_traits<float> >;
template class gen_bto_aux_symmetrize<4, bto_traits<float> >;
template class gen_bto_aux_symmetrize<5, bto_traits<float> >;
template class gen_bto_aux_symmetrize<6, bto_traits<float> >;
template class gen_bto_aux_symmetrize<7, bto_traits<float> >;
template class gen_bto_aux_symmetrize<8, bto_traits<float> >;

} // namespace libtensor
