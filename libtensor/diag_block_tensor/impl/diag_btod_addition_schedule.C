#include <libtensor/gen_block_tensor/impl/addition_schedule_impl.h>
#include <libtensor/diag_block_tensor/diag_btod_traits.h>

namespace libtensor {


template class addition_schedule<1, diag_btod_traits>;
template class addition_schedule<2, diag_btod_traits>;
template class addition_schedule<3, diag_btod_traits>;
template class addition_schedule<4, diag_btod_traits>;
template class addition_schedule<5, diag_btod_traits>;
template class addition_schedule<6, diag_btod_traits>;
template class addition_schedule<7, diag_btod_traits>;
template class addition_schedule<8, diag_btod_traits>;


} // namespace libtensor
