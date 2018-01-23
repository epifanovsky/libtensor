#include <libtensor/core/scalar_transf_double.h>
#include "to_btconv_impl.h"

namespace libtensor {


template class to_btconv<1, double>;
template class to_btconv<2, double>;
template class to_btconv<3, double>;
template class to_btconv<4, double>;
template class to_btconv<5, double>;
template class to_btconv<6, double>;
template class to_btconv<7, double>;
template class to_btconv<8, double>;

template class to_btconv<1, float>;
template class to_btconv<2, float>;
template class to_btconv<3, float>;
template class to_btconv<4, float>;
template class to_btconv<5, float>;
template class to_btconv<6, float>;
template class to_btconv<7, float>;
template class to_btconv<8, float>;

} // namespace libtensor
