#include <libtensor/linalg/linalg.h>
#include "kern_add2_impl.h"
#include "kern_add2_i_i_x_x_impl.h"
#include "kern_add2_i_x_i_x_impl.h"

namespace libtensor {


template class kern_add2<linalg, double>;
template class kern_add2<linalg, float>;


} // namespace libtensor
