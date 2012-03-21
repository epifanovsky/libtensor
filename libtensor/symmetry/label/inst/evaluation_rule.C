#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "../evaluation_rule.h"
#include "evaluation_rule_impl.h"


namespace libtensor {

template class evaluation_rule<1>;
template class evaluation_rule<2>;
template class evaluation_rule<3>;
template class evaluation_rule<4>;
template class evaluation_rule<5>;
template class evaluation_rule<6>;

} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

