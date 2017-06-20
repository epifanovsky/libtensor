#include "bto_symcontract3_impl.h"

namespace libtensor {


template class bto_symcontract3<1, 0, 1, 1, 1, double>;
template class bto_symcontract3<1, 1, 1, 1, 1, double>;
template class bto_symcontract3<2, 0, 2, 1, 2, double>;

template class bto_symcontract3<1, 0, 1, 1, 1, float>;
template class bto_symcontract3<1, 1, 1, 1, 1, float>;
template class bto_symcontract3<2, 0, 2, 1, 2, float>;

} // namespace libtensor

