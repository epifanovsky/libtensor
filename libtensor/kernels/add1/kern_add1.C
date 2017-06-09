#include <libtensor/linalg/linalg.h>
#include "kern_add1_impl.h"
#include "kern_add1_i_i_x_impl.h"
#include "kern_add1_ij_ij_x_impl.h"
#include "kern_add1_ij_ji_x_impl.h"

namespace libtensor {


template class kern_add1<linalg,double>;
//template class kern_add1<linalg,float>;


} // namespace libtensor
