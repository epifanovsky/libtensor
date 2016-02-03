#include <libtensor/gen_block_tensor/impl/gen_bto_trace_impl.h>
#include "ctf_btod_trace_impl.h"

namespace libtensor {


template class gen_bto_trace< 1, ctf_btod_traits, ctf_btod_trace<1> >;
template class gen_bto_trace< 2, ctf_btod_traits, ctf_btod_trace<2> >;
template class gen_bto_trace< 3, ctf_btod_traits, ctf_btod_trace<3> >;
template class gen_bto_trace< 4, ctf_btod_traits, ctf_btod_trace<4> >;

template class ctf_btod_trace<1>;
template class ctf_btod_trace<2>;
template class ctf_btod_trace<3>;
template class ctf_btod_trace<4>;


} // namespace libtensor
