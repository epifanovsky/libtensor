//#include <libtensor/core/allocator.h>
#include <libvmm/cuda_allocator.h>
#include <libtensor/core/scalar_transf_double.h>
//#include <libtensor/gen_block_tensor/impl/gen_block_tensor_impl.h>
#include <libtensor/gen_block_tensor/impl/block_map_impl.h>
#include "../cuda_block_tensor_traits.h"
#include "../cuda_block_tensor.h"

namespace libtensor {

typedef libvmm::cuda_allocator<double> cuda_allocator;

//typedef cuda_block_tensor_traits< double, libvmm::cuda_allocator<double> > cuda_bt_traits;

//template class cuda_block_tensor<1, cuda_bt_traits>;
//template class cuda_block_tensor<2, cuda_bt_traits>;
//template class cuda_block_tensor<3, cuda_bt_traits>;
//template class cuda_block_tensor<4, cuda_bt_traits>;
//template class cuda_block_tensor<5, cuda_bt_traits>;
//template class cuda_block_tensor<6, cuda_bt_traits>;
//template class cuda_block_tensor<7, cuda_bt_traits>;
//template class cuda_block_tensor<8, cuda_bt_traits>;

template class cuda_block_tensor<1, double, cuda_allocator>;
template class cuda_block_tensor<2, double , cuda_allocator>;
template class cuda_block_tensor<3, double , cuda_allocator>;
template class cuda_block_tensor<4, double , cuda_allocator>;
template class cuda_block_tensor<5, double , cuda_allocator>;
template class cuda_block_tensor<6, double , cuda_allocator>;
template class cuda_block_tensor<7, double , cuda_allocator>;
template class cuda_block_tensor<8, double , cuda_allocator>;

//typedef cuda_block_tensor_traits< double, allocator<double> > bt_traits;
//
//template class gen_block_tensor<1, bt_traits>;
//template class gen_block_tensor<2, bt_traits>;
//template class gen_block_tensor<3, bt_traits>;
//template class gen_block_tensor<4, bt_traits>;
//template class gen_block_tensor<5, bt_traits>;
//template class gen_block_tensor<6, bt_traits>;
//template class gen_block_tensor<7, bt_traits>;
//template class gen_block_tensor<8, bt_traits>;


} // namespace libtensor
