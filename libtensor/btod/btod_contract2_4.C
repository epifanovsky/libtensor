#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "btod_contract2.h"
#include "btod_contract2_impl.h"

namespace libtensor {


template class btod_contract2<0, 4, 1>;
template class btod_contract2<0, 4, 2>;
template class btod_contract2<1, 3, 0>;
template class btod_contract2<1, 3, 1>;
template class btod_contract2<1, 3, 2>;
template class btod_contract2<1, 3, 3>;
template class btod_contract2<2, 2, 0>;
template class btod_contract2<2, 2, 1>;
template class btod_contract2<2, 2, 2>;
template class btod_contract2<2, 2, 3>;
template class btod_contract2<2, 2, 4>;
template class btod_contract2<3, 1, 0>;
template class btod_contract2<3, 1, 1>;
template class btod_contract2<3, 1, 2>;
template class btod_contract2<3, 1, 3>;
template class btod_contract2<4, 0, 1>;
template class btod_contract2<4, 0, 2>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
