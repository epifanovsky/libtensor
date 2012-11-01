#include <libtensor/linalg/cublas/linalg_cublas.h>
#include "kern_dadd1_impl.h"
#include "kern_dadd1_i_i_x_impl.h"
#include "kern_dadd1_ij_ij_x_impl.h"
#include "kern_dadd1_ij_ji_x_impl.h"

namespace libtensor {


template class kern_dadd1<linalg_cublas>;


} // namespace libtensor
