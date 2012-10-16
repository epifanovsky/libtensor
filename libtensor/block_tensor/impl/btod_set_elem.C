#include <libtensor/gen_block_tensor/impl/gen_bto_set_elem_impl.h>
#include "btod_set_elem_impl.h"

namespace libtensor {


template class gen_bto_set_elem<1, btod_traits>;
template class gen_bto_set_elem<2, btod_traits>;
template class gen_bto_set_elem<3, btod_traits>;
template class gen_bto_set_elem<4, btod_traits>;
template class gen_bto_set_elem<5, btod_traits>;
template class gen_bto_set_elem<6, btod_traits>;
template class gen_bto_set_elem<7, btod_traits>;
template class gen_bto_set_elem<8, btod_traits>;

template class btod_set_elem<1>;
template class btod_set_elem<2>;
template class btod_set_elem<3>;
template class btod_set_elem<4>;
template class btod_set_elem<5>;
template class btod_set_elem<6>;
template class btod_set_elem<7>;
template class btod_set_elem<8>;


} // namespace libtensor
