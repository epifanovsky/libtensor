#include <libtensor/core/scalar_transf_double.h>
#include "btod_extract_impl.h"

namespace libtensor {


template class btod_extract<2, 1>;

template class btod_extract<3, 1>;
template class btod_extract<3, 2>;

template class btod_extract<4, 1>;
template class btod_extract<4, 2>;
template class btod_extract<4, 3>;

template class btod_extract<5, 1>;
template class btod_extract<5, 2>;
template class btod_extract<5, 3>;
template class btod_extract<5, 4>;

template class btod_extract<6, 1>;
template class btod_extract<6, 2>;
template class btod_extract<6, 3>;
template class btod_extract<6, 4>;
template class btod_extract<6, 5>;


} // namespace libtensor

