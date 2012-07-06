#include "dimensions.h"

namespace libtensor {


const char *dimensions<0>::k_clazz = "dimensions<0>";


} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "dimensions.h"
#include "dimensions_impl.h"

namespace libtensor {


template class dimensions<1>;
template class dimensions<2>;
template class dimensions<3>;
template class dimensions<4>;
template class dimensions<5>;
template class dimensions<6>;
template class dimensions<7>;
template class dimensions<8>;
template class dimensions<9>;
template class dimensions<10>;
template class dimensions<11>;
template class dimensions<12>;
template class dimensions<13>;
template class dimensions<14>;
template class dimensions<15>;
template class dimensions<16>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
