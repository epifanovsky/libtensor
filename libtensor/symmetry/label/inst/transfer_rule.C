#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "../transfer_rule.h"
#include "transfer_rule_impl.h"


namespace libtensor {

template class transfer_rule<1>;
template class transfer_rule<2>;
template class transfer_rule<3>;
template class transfer_rule<4>;
template class transfer_rule<5>;
template class transfer_rule<6>;

} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

