#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include <libtensor/core/scalar_transf_double.h>
#include "btod_dotprod.h"
#include "btod_dotprod_impl.h"

namespace libtensor {


template class btod_dotprod<1>;
template class btod_dotprod<2>;
template class btod_dotprod<3>;
template class btod_dotprod<4>;
template class btod_dotprod<5>;
template class btod_dotprod<6>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
