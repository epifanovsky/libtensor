#ifndef CTF_TOD_COLLECT_IMPL_H
#define CTF_TOD_COLLECT_IMPL_H

#include <cstring>
#include <vector>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_collect.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_collect<N>::k_clazz[] = "ctf_tod_collect<N>";


template<size_t N>
void ctf_tod_collect<N>::perform(dense_tensor_wr_i<N, double> &t) {

    static const char method[] = "perform(dense_tensor_wr_i<N, double>&)";

    if(!m_dt.get_dims().equals(t.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    size_t sz = m_dt.get_dims().get_size();

    ctf_dense_tensor_ctrl<N, double> dctrl(m_dt);
    dense_tensor_wr_ctrl<N, double> ctrl(t);

    tCTF_Tensor<double> &dt = dctrl.req_ctf_tensor();
    double *p = ctrl.req_dataptr();

    ::memset(p, 0, sizeof(double) * sz);

    const size_t bufsz = 1024 * 1024;
    std::vector<long_int> idx(bufsz);
    for(size_t i = 0; i < sz; i += bufsz) {
        size_t n = std::min(sz - i, bufsz);
        for(size_t j = 0; j < n; j++) idx[j] = long_int(i + j);
        dt.read(n, &idx[0], p + i);
    }

    ctrl.ret_dataptr(p);
}


} // namespace libtensor

#endif // CTF_TOD_COLLECT_IMPL_H

