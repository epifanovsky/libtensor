#include <libtensor/gen_block_tensor/impl/gen_bto_aux_chsym_impl.h>
#include <libtensor/block_tensor/bto_traits.h>

namespace libtensor {


template class gen_bto_aux_chsym<1, bto_traits<double> >;
template class gen_bto_aux_chsym<2, bto_traits<double> >;
template class gen_bto_aux_chsym<3, bto_traits<double> >;
template class gen_bto_aux_chsym<4, bto_traits<double> >;
template class gen_bto_aux_chsym<5, bto_traits<double> >;
template class gen_bto_aux_chsym<6, bto_traits<double> >;
template class gen_bto_aux_chsym<7, bto_traits<double> >;
template class gen_bto_aux_chsym<8, bto_traits<double> >;

template class gen_bto_aux_chsym<1, bto_traits<float> >;
template class gen_bto_aux_chsym<2, bto_traits<float> >;
template class gen_bto_aux_chsym<3, bto_traits<float> >;
template class gen_bto_aux_chsym<4, bto_traits<float> >;
template class gen_bto_aux_chsym<5, bto_traits<float> >;
template class gen_bto_aux_chsym<6, bto_traits<float> >;
template class gen_bto_aux_chsym<7, bto_traits<float> >;
template class gen_bto_aux_chsym<8, bto_traits<float> >;

} // namespace libtensor
