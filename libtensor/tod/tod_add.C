#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "tod_add.h"
#include "tod_add_impl.h"

namespace libtensor {


template class tod_add<1>;
template class tod_add<2>;
template class tod_add<3>;
template class tod_add<4>;
template class tod_add<5>;
template class tod_add<6>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
