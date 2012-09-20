#include <libtensor/core/allocator.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/gen_block_tensor/impl/gen_block_tensor_impl.h>
#include <libtensor/gen_block_tensor/impl/block_map_impl.h>
#include "../diag_block_tensor_traits.h"

namespace libtensor {


typedef diag_block_tensor_traits< double, std_allocator<double> > std_bt_traits;

template class gen_block_tensor<1, std_bt_traits>;
template class gen_block_tensor<2, std_bt_traits>;
template class gen_block_tensor<3, std_bt_traits>;
template class gen_block_tensor<4, std_bt_traits>;
template class gen_block_tensor<5, std_bt_traits>;
template class gen_block_tensor<6, std_bt_traits>;
template class gen_block_tensor<7, std_bt_traits>;
template class gen_block_tensor<8, std_bt_traits>;

typedef diag_block_tensor_traits< double, allocator<double> > bt_traits;

template class gen_block_tensor<1, bt_traits>;
template class gen_block_tensor<2, bt_traits>;
template class gen_block_tensor<3, bt_traits>;
template class gen_block_tensor<4, bt_traits>;
template class gen_block_tensor<5, bt_traits>;
template class gen_block_tensor<6, bt_traits>;
template class gen_block_tensor<7, bt_traits>;
template class gen_block_tensor<8, bt_traits>;


} // namespace libtensor
