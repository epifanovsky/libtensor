#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include <libtensor/core/scalar_transf_double.h>
#include "btod_sum.h"
#include "btod_sum_impl.h"

namespace libtensor {


template class btod_sum<1>;
template class btod_sum<2>;
template class btod_sum<3>;
template class btod_sum<4>;
template class btod_sum<5>;
template class btod_sum<6>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
