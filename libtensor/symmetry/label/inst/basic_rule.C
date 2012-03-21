#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

#include "../basic_rule.h"
#include "basic_rule_impl.h"


namespace libtensor {

template class basic_rule<1>;
template class basic_rule<2>;
template class basic_rule<3>;
template class basic_rule<4>;
template class basic_rule<5>;
template class basic_rule<6>;

template bool operator==(const basic_rule<1> &, const basic_rule<1> &);
template bool operator==(const basic_rule<2> &, const basic_rule<2> &);
template bool operator==(const basic_rule<3> &, const basic_rule<3> &);
template bool operator==(const basic_rule<4> &, const basic_rule<4> &);
template bool operator==(const basic_rule<5> &, const basic_rule<5> &);
template bool operator==(const basic_rule<6> &, const basic_rule<6> &);

} // namespace libtensor

#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

