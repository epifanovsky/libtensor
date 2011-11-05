#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "orbit_list.h"
#include "../btod/transf_double.h"
#include "orbit_list_impl.h"

namespace libtensor {


template class orbit_list<1, double>;
template class orbit_list<2, double>;
template class orbit_list<3, double>;
template class orbit_list<4, double>;
template class orbit_list<5, double>;
template class orbit_list<6, double>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
