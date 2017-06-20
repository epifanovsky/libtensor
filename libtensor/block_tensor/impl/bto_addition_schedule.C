#include <libtensor/gen_block_tensor/impl/addition_schedule_impl.h>
#include <libtensor/block_tensor/bto_traits.h>

namespace libtensor {


template class addition_schedule<1, bto_traits<double> >;
template class addition_schedule<2, bto_traits<double> >;
template class addition_schedule<3, bto_traits<double> >;
template class addition_schedule<4, bto_traits<double> >;
template class addition_schedule<5, bto_traits<double> >;
template class addition_schedule<6, bto_traits<double> >;
template class addition_schedule<7, bto_traits<double> >;
template class addition_schedule<8, bto_traits<double> >;

template class addition_schedule<1, bto_traits<float> >;
template class addition_schedule<2, bto_traits<float> >;
template class addition_schedule<3, bto_traits<float> >;
template class addition_schedule<4, bto_traits<float> >;
template class addition_schedule<5, bto_traits<float> >;
template class addition_schedule<6, bto_traits<float> >;
template class addition_schedule<7, bto_traits<float> >;
template class addition_schedule<8, bto_traits<float> >;

} // namespace libtensor
