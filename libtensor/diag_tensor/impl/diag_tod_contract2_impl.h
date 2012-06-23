#ifndef LIBTENSOR_DIAG_TOD_CONTRACT2_IMPL_H
#define LIBTENSOR_DIAG_TOD_CONTRACT2_IMPL_H

#include "../diag_tod_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *diag_tod_contract2<N, M, K>::k_clazz =
    "diag_tod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
void diag_tod_contract2<N, M, K>::perform(
    diag_tensor_wr_i<N + M, double> &dtc) {

}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_CONTRACT2_IMPL_H
