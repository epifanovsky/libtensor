#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "btod_contract2.h"
#include "btod_contract2_impl.h"

namespace libtensor {


template class btod_contract2<1, 5, 0>;
template class btod_contract2<1, 5, 1>;
template class btod_contract2<2, 4, 0>;
template class btod_contract2<2, 4, 1>;
template class btod_contract2<2, 4, 2>;
template class btod_contract2<3, 3, 0>;
template class btod_contract2<3, 3, 1>;
template class btod_contract2<3, 3, 2>;
template class btod_contract2<3, 3, 3>;
template class btod_contract2<4, 2, 0>;
template class btod_contract2<4, 2, 1>;
template class btod_contract2<4, 2, 2>;
template class btod_contract2<5, 1, 0>;
template class btod_contract2<5, 1, 1>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
