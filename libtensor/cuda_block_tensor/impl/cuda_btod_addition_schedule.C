#include <libtensor/gen_block_tensor/impl/addition_schedule_impl.h>
#include <libtensor/cuda_block_tensor/cuda_btod_traits.h>

namespace libtensor {


template class addition_schedule<1, cuda_btod_traits>;
template class addition_schedule<2, cuda_btod_traits>;
template class addition_schedule<3, cuda_btod_traits>;
template class addition_schedule<4, cuda_btod_traits>;
template class addition_schedule<5, cuda_btod_traits>;
template class addition_schedule<6, cuda_btod_traits>;


} // namespace libtensor
