#include <libtensor/block_tensor/bto/bto_trace.h>
#include <libtensor/block_tensor/bto/impl/bto_trace_impl.h>
#include <libtensor/btod/btod_trace.h>

namespace libtensor {


template class bto_trace<1, btod_trace_traits>;
template class bto_trace<2, btod_trace_traits>;
template class bto_trace<3, btod_trace_traits>;


template class btod_trace<1>;
template class btod_trace<2>;
template class btod_trace<3>;


} // namespace libtensor
