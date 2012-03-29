#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "scalar_transf_double.h"
#include "additive_btod.h"
#include "additive_btod_impl.h"

namespace libtensor {


template class additive_btod<1>;
template class additive_btod<2>;
template class additive_btod<3>;
template class additive_btod<4>;
template class additive_btod<5>;
template class additive_btod<6>;


} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES
