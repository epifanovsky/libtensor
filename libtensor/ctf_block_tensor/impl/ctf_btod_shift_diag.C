#include <libtensor/gen_block_tensor/impl/gen_bto_shift_diag_impl.h>
#include "ctf_btod_shift_diag_impl.h"

namespace libtensor {


template class gen_bto_shift_diag<1, ctf_btod_traits>;
template class gen_bto_shift_diag<2, ctf_btod_traits>;
template class gen_bto_shift_diag<3, ctf_btod_traits>;
template class gen_bto_shift_diag<4, ctf_btod_traits>;
template class gen_bto_shift_diag<5, ctf_btod_traits>;
template class gen_bto_shift_diag<6, ctf_btod_traits>;
template class gen_bto_shift_diag<7, ctf_btod_traits>;
template class gen_bto_shift_diag<8, ctf_btod_traits>;

template class ctf_btod_shift_diag<1>;
template class ctf_btod_shift_diag<2>;
template class ctf_btod_shift_diag<3>;
template class ctf_btod_shift_diag<4>;
template class ctf_btod_shift_diag<5>;
template class ctf_btod_shift_diag<6>;
template class ctf_btod_shift_diag<7>;
template class ctf_btod_shift_diag<8>;


} // namespace libtensor
