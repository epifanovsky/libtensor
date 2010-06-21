#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "btod_contract2.h"
#include "btod_contract2_impl.h"

namespace libtensor {


template class btod_contract2<0, 1, 1>;
template class btod_contract2<0, 1, 2>;
template class btod_contract2<0, 1, 3>;
template class btod_contract2<0, 1, 4>;
template class btod_contract2<0, 1, 5>;
template class btod_contract2<1, 0, 1>;
template class btod_contract2<1, 0, 2>;
template class btod_contract2<1, 0, 3>;
template class btod_contract2<1, 0, 4>;
template class btod_contract2<1, 0, 5>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
