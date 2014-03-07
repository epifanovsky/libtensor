#ifndef CTF_TOD_DISTRIBUTE_IMPL_H
#define CTF_TOD_DISTRIBUTE_IMPL_H

#include <vector>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../ctf.h"
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_tod_distribute.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_distribute<N>::k_clazz[] = "ctf_tod_distribute<N>";


template<size_t N>
void ctf_tod_distribute<N>::perform(ctf_dense_tensor_i<N, double> &dt) {

    static const char method[] = "perform(ctf_dense_tensor_i<N, double>&)";

    if(!m_t.get_dims().equals(dt.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "dt");
    }

    size_t sz = m_t.get_dims().get_size();
    std::vector<long_int> keys;
    std::vector<double> data;

    if(ctf::is_master()) {

        data.reserve(sz);

        dense_tensor_rd_ctrl<N, double> ctrl(m_t);
        const double *p = ctrl.req_const_dataptr();
        for(size_t i = 0; i < sz; i++) {
            keys.push_back(i);
            data.push_back(p[i]);
        }
        ctrl.ret_const_dataptr(p);

        ctf_dense_tensor_ctrl<N, double> dctrl(dt);
        tCTF_Tensor<double> &dt = dctrl.req_ctf_tensor();
        dt.write_remote_data(sz, &keys[0], &data[0]);

    } else {

        ctf_dense_tensor_ctrl<N, double> dctrl(dt);
        tCTF_Tensor<double> &dt = dctrl.req_ctf_tensor();
        dt.write_remote_data(0, 0, 0);

    }
}


} // namespace libtensor

#endif // CTF_TOD_DISTRIBUTE_IMPL_H

