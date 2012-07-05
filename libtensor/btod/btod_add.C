#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include <libtensor/core/scalar_transf_double.h>
#include "btod_add.h"
#include "btod_add_impl.h"

namespace libtensor {


template class btod_add<1>;
template class btod_add<2>;
template class btod_add<3>;
template class btod_add<4>;
template class btod_add<5>;
template class btod_add<6>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
