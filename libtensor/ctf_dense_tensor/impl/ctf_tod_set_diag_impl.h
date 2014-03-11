#ifndef LIBTENSOR_CTF_TOD_SET_DIAG_IMPL_H
#define LIBTENSOR_CTF_TOD_SET_DIAG_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_tod_set_diag.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_set_diag<N>::k_clazz[] = "ctf_tod_set_diag<N>";


template<size_t N>
void ctf_tod_set_diag<N>::perform(ctf_dense_tensor_i<N, double> &ta) {

    static const char method[] = "perform(ctf_dense_tensor_i<N, double>&)";

    const dimensions<N> &dims = ta.get_dims();
    size_t n = dims[0];
    for(size_t i = 1; i < N; i++) {
        if(dims[i] != n) {
            throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
        }
    }

    size_t inc = 0;
    for(size_t i = 0; i < N; i++) inc += dims.get_increment(i);

    ctf_dense_tensor_ctrl<N, double> ca(ta);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();

    const size_t bufsz = 1024 * 1024;
    std::vector<long_int> idx(bufsz);
    std::vector<double> data(bufsz, m_v);
    for(size_t i = 0; i < n; i += bufsz) {
        size_t m = std::min(n - i, bufsz);
        for(size_t j = 0; j < m; j++) idx[j] = long_int((i + j) * inc);
        if(ctf::is_master()) {
            dta.write(m, &idx[0], &data[0]);
        } else {
            dta.write(0, 0, 0);
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SET_DIAG_IMPL_H
