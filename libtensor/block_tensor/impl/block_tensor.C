#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_block_tensor_impl.h>
#include <libtensor/gen_block_tensor/impl/block_map_impl.h>
#include "../block_tensor_traits.h"

namespace libtensor {


//typedef block_tensor_traits< double, allocator<double> > bt_traits;

template class gen_block_tensor<1, block_tensor_traits< double, allocator > >;
template class gen_block_tensor<2, block_tensor_traits< double, allocator > >;
template class gen_block_tensor<3, block_tensor_traits< double, allocator > >;
template class gen_block_tensor<4, block_tensor_traits< double, allocator > >;
template class gen_block_tensor<5, block_tensor_traits< double, allocator > >;
template class gen_block_tensor<6, block_tensor_traits< double, allocator > >;
template class gen_block_tensor<7, block_tensor_traits< double, allocator > >;
template class gen_block_tensor<8, block_tensor_traits< double, allocator > >;

template class gen_block_tensor<1, block_tensor_traits< float, allocator > >;
template class gen_block_tensor<2, block_tensor_traits< float, allocator > >;
template class gen_block_tensor<3, block_tensor_traits< float, allocator > >;
template class gen_block_tensor<4, block_tensor_traits< float, allocator > >;
template class gen_block_tensor<5, block_tensor_traits< float, allocator > >;
template class gen_block_tensor<6, block_tensor_traits< float, allocator > >;
template class gen_block_tensor<7, block_tensor_traits< float, allocator > >;
template class gen_block_tensor<8, block_tensor_traits< float, allocator > >;

} // namespace libtensor
