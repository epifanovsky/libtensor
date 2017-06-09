#include <libtensor/linalg/linalg.h>
#include "kern_copy_impl.h"
#include "kern_copy_i_i_x_impl.h"
#include "kern_copy_ij_ij_x_impl.h"
#include "kern_copy_ij_ji_x_impl.h"

namespace libtensor {


template class kern_copy<linalg,double>;
//template class kern_copy<linalg,float>; // disabled to desing the structure first


} // namespace libtensor
