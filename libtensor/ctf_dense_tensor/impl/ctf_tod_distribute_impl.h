#ifndef CTF_TOD_DISTRIBUTE_IMPL_H
#define CTF_TOD_DISTRIBUTE_IMPL_H

#include <vector>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../ctf.h"
#include "../ctf_dense_tensor.h"
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

    long_int np, *idx;
    double *pdst;

    ctf_dense_tensor_ctrl<N, double> dctrl(dt);
    CTF::Tensor<double> &dta = dctrl.req_ctf_tensor();
    dta.read_local(&np, &idx, &pdst);

    dense_tensor_rd_ctrl<N, double> ctrl(m_t);
    const double *psrc = ctrl.req_const_dataptr();
    for(size_t i = 0; i < np; i++) {
        pdst[i] = psrc[idx[i]];
    }
    ctrl.ret_const_dataptr(psrc);

    dta.write(np, idx, pdst);

    free(idx);
    free(pdst);
}


} // namespace libtensor

#endif // CTF_TOD_DISTRIBUTE_IMPL_H

