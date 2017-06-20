#include "bto_contract3_impl.h"

namespace libtensor {


template class bto_contract3<1, 0, 1, 1, 1, double>;
template class bto_contract3<1, 1, 1, 1, 1, double>;
template class bto_contract3<1, 1, 2, 0, 0, double>;
template class bto_contract3<1, 1, 2, 1, 2, double>;
template class bto_contract3<2, 0, 0, 0, 2, double>;
template class bto_contract3<2, 0, 1, 1, 2, double>;
template class bto_contract3<2, 0, 2, 1, 2, double>;

template class bto_contract3<1, 0, 1, 1, 1, float>;
template class bto_contract3<1, 1, 1, 1, 1, float>;
template class bto_contract3<1, 1, 2, 0, 0, float>;
template class bto_contract3<1, 1, 2, 1, 2, float>;
template class bto_contract3<2, 0, 0, 0, 2, float>;
template class bto_contract3<2, 0, 1, 1, 2, float>;
template class bto_contract3<2, 0, 2, 1, 2, float>;

} // namespace libtensor

