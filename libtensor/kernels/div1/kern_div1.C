#include <libtensor/linalg/linalg.h>
#include "kern_div1_impl.h"
#include "kern_div1_i_i_x_impl.h"

namespace libtensor {


template class kern_div1<linalg, double>;
template class kern_div1<linalg, float>;


} // namespace libtensor
