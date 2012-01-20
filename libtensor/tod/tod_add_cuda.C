#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "tod_add_cuda.h"
#include "tod_add_cuda_impl.h"

namespace libtensor {


template class tod_add_cuda<1>;
template class tod_add_cuda<2>;
template class tod_add_cuda<3>;
template class tod_add_cuda<4>;
template class tod_add_cuda<5>;
template class tod_add_cuda<6>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
