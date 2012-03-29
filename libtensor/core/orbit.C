#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "../btod/scalar_transf_double.h"
#include "orbit.h"
#include "orbit_impl.h"

namespace libtensor {


template class orbit<1, double>;
template class orbit<2, double>;
template class orbit<3, double>;
template class orbit<4, double>;
template class orbit<5, double>;
template class orbit<6, double>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
