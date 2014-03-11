#ifndef LIBTENSOR_CTF_TOD_SET_IMPL_H
#define LIBTENSOR_CTF_TOD_SET_IMPL_H

#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_tod_set.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_set<N>::k_clazz[] = "ctf_tod_set<N>";


template<size_t N>
void ctf_tod_set<N>::perform(ctf_dense_tensor_i<N, double> &ta) {

    size_t sz = ta.get_dims().get_size();

    ctf_dense_tensor_ctrl<N, double> ca(ta);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();

    const size_t bufsz = 1024 * 1024;
    std::vector<long_int> idx(bufsz);
    std::vector<double> data(bufsz, m_v);
    for(size_t i = 0; i < sz; i += bufsz) {
        size_t n = std::min(sz - i, bufsz);
        for(size_t j = 0; j < n; j++) idx[j] = long_int(i + j);
        if(ctf::is_master()) {
            dta.write(n, &idx[0], &data[0]);
        } else {
            dta.write(0, 0, 0);
        }
    }

}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SET_IMPL_H
