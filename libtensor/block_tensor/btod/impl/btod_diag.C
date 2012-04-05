#include <libtensor/block_tensor/bto/bto_diag.h>
#include <libtensor/block_tensor/bto/impl/bto_diag_impl.h>
#include <libtensor/btod/btod_diag.h>

namespace libtensor {


template class bto_diag<2, 2, btod_diag_traits>;
template class bto_diag<3, 2, btod_diag_traits>;
template class bto_diag<3, 3, btod_diag_traits>;
template class bto_diag<4, 2, btod_diag_traits>;
template class bto_diag<4, 3, btod_diag_traits>;
template class bto_diag<4, 4, btod_diag_traits>;
template class bto_diag<5, 2, btod_diag_traits>;
template class bto_diag<5, 3, btod_diag_traits>;
template class bto_diag<5, 4, btod_diag_traits>;
template class bto_diag<5, 5, btod_diag_traits>;
template class bto_diag<6, 2, btod_diag_traits>;
template class bto_diag<6, 3, btod_diag_traits>;
template class bto_diag<6, 4, btod_diag_traits>;
template class bto_diag<6, 5, btod_diag_traits>;
template class bto_diag<6, 6, btod_diag_traits>;


template class btod_diag<2, 2>;
template class btod_diag<3, 2>;
template class btod_diag<3, 3>;
template class btod_diag<4, 2>;
template class btod_diag<4, 3>;
template class btod_diag<4, 4>;
template class btod_diag<5, 2>;
template class btod_diag<5, 3>;
template class btod_diag<5, 4>;
template class btod_diag<5, 5>;
template class btod_diag<6, 2>;
template class btod_diag<6, 3>;
template class btod_diag<6, 4>;
template class btod_diag<6, 5>;
template class btod_diag<6, 6>;


} // namespace libtensor

