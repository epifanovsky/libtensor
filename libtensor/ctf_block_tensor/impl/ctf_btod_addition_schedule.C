#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/addition_schedule_impl.h>
#include <libtensor/ctf_block_tensor/ctf_btod_traits.h>

namespace libtensor {


template class addition_schedule<1, ctf_btod_traits>;
template class addition_schedule<2, ctf_btod_traits>;
template class addition_schedule<3, ctf_btod_traits>;
template class addition_schedule<4, ctf_btod_traits>;
template class addition_schedule<5, ctf_btod_traits>;
template class addition_schedule<6, ctf_btod_traits>;
template class addition_schedule<7, ctf_btod_traits>;
template class addition_schedule<8, ctf_btod_traits>;


} // namespace libtensor
