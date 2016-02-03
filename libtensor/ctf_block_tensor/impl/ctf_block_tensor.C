#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_block_tensor_impl.h>
#include <libtensor/gen_block_tensor/impl/block_map_impl.h>
#include "../ctf_block_tensor_traits.h"

namespace libtensor {


typedef ctf_block_tensor_traits<double> bt_traits;

template class gen_block_tensor<1, bt_traits>;
template class gen_block_tensor<2, bt_traits>;
template class gen_block_tensor<3, bt_traits>;
template class gen_block_tensor<4, bt_traits>;
template class gen_block_tensor<5, bt_traits>;
template class gen_block_tensor<6, bt_traits>;
template class gen_block_tensor<7, bt_traits>;
template class gen_block_tensor<8, bt_traits>;


} // namespace libtensor
