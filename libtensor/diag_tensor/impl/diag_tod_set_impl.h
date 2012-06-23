#include <vector>
#include "../diag_tensor_ctrl.h"
#include "../diag_tod_set.h"

namespace libtensor {


template<size_t N>
const char *diag_tod_set<N>::k_clazz = "diag_tod_set<N>";


template<size_t N>
void diag_tod_set<N>::perform(diag_tensor_wr_i<N, double> &ta) {

    diag_tod_set<N>::start_timer();

    try {

        diag_tensor_wr_ctrl<N, double> ca(ta);

        const diag_tensor_space<N> &dtsa = ta.get_space();
        std::vector<size_t> ssl;
        dtsa.get_all_subspaces(ssl);
        for(size_t ssi = 0; ssi < ssl.size(); ssi++) {

            size_t ssn = ssl[ssi];
            size_t sz = dtsa.get_subspace_size(ssn);
            double *pa = ca.req_dataptr(ssn);
            for(size_t i = 0; i < sz; i++) pa[i] = m_d;
            ca.ret_dataptr(ssn, pa);
        }

    } catch(...) {
        diag_tod_set<N>::stop_timer();
        throw;
    }

    diag_tod_set<N>::stop_timer();
}


} // namespace libtensor

