#ifndef CTF_TOD_COLLECT_IMPL_H
#define CTF_TOD_COLLECT_IMPL_H

#include <cstring>
#include <vector>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../ctf.h"
#include "../ctf_dense_tensor_ctrl.h"
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
    double *data;
    int64_t n;

    ctf_dense_tensor_ctrl<N, double> dctrl(m_dt);
    int tid = dctrl.req_tensor_id();
    ctf::get().allread_tensor(tid, &n, &data);

    if(n != sz) {
        //free_buffer_space(data);
        free(data);
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "dt");
    }

    dense_tensor_wr_ctrl<N, double> ctrl(t);
    double *p = ctrl.req_dataptr();
    ::memcpy(p, data, sz * sizeof(double));
    ctrl.ret_dataptr(p);

    //free_buffer_space(data);
    free(data);
}


} // namespace libtensor

#endif // CTF_TOD_COLLECT_IMPL_H

