#ifndef LIBTENSOR_DIAG_TOD_RANDOM_IMPL_H
#define LIBTENSOR_DIAG_TOD_RANDOM_IMPL_H

#include <libtensor/linalg/linalg.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include "../diag_tod_random.h"

namespace libtensor {


template<size_t N>
const char *diag_tod_random<N>::k_clazz = "diag_tod_random<N>";


template<size_t N>
void diag_tod_random<N>::perform(diag_tensor_wr_i<N, double> &ta) {

    diag_tod_random::start_timer();

    try {

        diag_tensor_wr_ctrl<N, double> ca(ta);

        const diag_tensor_space<N> &dtsa = ta.get_space();
        std::vector<size_t> ssl;
        dtsa.get_all_subspaces(ssl);
        for(size_t ssi = 0; ssi < ssl.size(); ssi++) {

            size_t ssn = ssl[ssi];
            size_t sz = dtsa.get_subspace_size(ssn);
            double *pa = ca.req_dataptr(ssn);
            linalg::rng_set_i_x(0, sz, pa, 1, 1.0);
            ca.ret_dataptr(ssn, pa);
        }

    } catch(...) {
        diag_tod_random::stop_timer();
        throw;
    }

    diag_tod_random::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_RANDOM_IMPL_H
