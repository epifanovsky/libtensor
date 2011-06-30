#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "tod_dirsum.h"
#include "tod_dirsum_impl.h"

namespace libtensor {


template class tod_dirsum<1, 1>;

template class tod_dirsum<1, 2>;
template class tod_dirsum<2, 1>;

template class tod_dirsum<1, 3>;
template class tod_dirsum<2, 2>;
template class tod_dirsum<3, 1>;

template class tod_dirsum<1, 4>;
template class tod_dirsum<2, 3>;
template class tod_dirsum<3, 2>;
template class tod_dirsum<4, 1>;

template class tod_dirsum<1, 5>;
template class tod_dirsum<2, 4>;
template class tod_dirsum<3, 3>;
template class tod_dirsum<4, 2>;
template class tod_dirsum<5, 1>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
