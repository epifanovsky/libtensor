#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "../se_label.h"
#include "se_label_impl.h"


namespace libtensor {

template class se_label<1, double>;
template class se_label<2, double>;
template class se_label<3, double>;
template class se_label<4, double>;
template class se_label<5, double>;
template class se_label<6, double>;

} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

