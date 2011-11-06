#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "tod_dotprod.h"
#include "tod_dotprod_impl.h"

namespace libtensor {


template class tod_dotprod<1>;
template class tod_dotprod<2>;
template class tod_dotprod<3>;
template class tod_dotprod<4>;
template class tod_dotprod<5>;
template class tod_dotprod<6>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
