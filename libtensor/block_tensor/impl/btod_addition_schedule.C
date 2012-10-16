#include <libtensor/gen_block_tensor/impl/addition_schedule_impl.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


template class addition_schedule<1, btod_traits>;
template class addition_schedule<2, btod_traits>;
template class addition_schedule<3, btod_traits>;
template class addition_schedule<4, btod_traits>;
template class addition_schedule<5, btod_traits>;
template class addition_schedule<6, btod_traits>;
template class addition_schedule<7, btod_traits>;
template class addition_schedule<8, btod_traits>;


} // namespace libtensor
