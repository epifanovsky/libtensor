#include <libtensor/gen_block_tensor/impl/gen_bto_trace_impl.h>
#include "../btod_trace.h"

namespace libtensor {


template<size_t N>
const char *btod_trace<N>::k_clazz = "btod_trace<N>";

template class gen_bto_trace< 1, btod_traits, btod_trace<1> >;
template class gen_bto_trace< 2, btod_traits, btod_trace<2> >;
template class gen_bto_trace< 3, btod_traits, btod_trace<3> >;

template class btod_trace<1>;
template class btod_trace<2>;
template class btod_trace<3>;


} // namespace libtensor
