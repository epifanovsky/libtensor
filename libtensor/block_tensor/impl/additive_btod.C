#include <libtensor/block_tensor/block_tensor_i_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/additive_gen_bto_impl.h>

namespace libtensor {


template class additive_gen_bto<1, block_tensor_i_traits<double> >;
template class additive_gen_bto<2, block_tensor_i_traits<double> >;
template class additive_gen_bto<3, block_tensor_i_traits<double> >;
template class additive_gen_bto<4, block_tensor_i_traits<double> >;
template class additive_gen_bto<5, block_tensor_i_traits<double> >;
template class additive_gen_bto<6, block_tensor_i_traits<double> >;
template class additive_gen_bto<7, block_tensor_i_traits<double> >;
template class additive_gen_bto<8, block_tensor_i_traits<double> >;


} // namespace libtensor
