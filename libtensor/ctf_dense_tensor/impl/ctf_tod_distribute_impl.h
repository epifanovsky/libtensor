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

    sequence<N, unsigned> symt_grp, symt_sym;
    for(size_t i = 0; i < N; i++) symt_grp[i] = i;
    ctf_symmetry<N, double> symt(symt_grp, symt_sym);
    ctf_dense_tensor<N, double> dtmp(dt.get_dims(), symt);
    ctf_dense_tensor_ctrl<N, double> dct(dtmp);
    CTF::Tensor<double> &dtt = dct.req_ctf_tensor(0);
    dtt.read_local(&np, &idx, &pdst);
    dense_tensor_rd_ctrl<N, double> ctrl(m_t);
    const double *psrc = ctrl.req_const_dataptr();
    for(size_t i = 0; i < np; i++) {
        pdst[i] = psrc[idx[i]];
    }
    ctrl.ret_const_dataptr(psrc);
    dtt.write(np, idx, pdst);
    free(idx);
    free(pdst);

    ctf_dense_tensor_ctrl<N, double> dca(dt);
    const ctf_symmetry<N, double> &syma = dca.req_symmetry();
    char labela[N + 1];
    for(size_t i = 0; i < N; i++) labela[i] = char(i) + 1;
    labela[N] = '\0';

    for(size_t icomp = 0; icomp < syma.get_ncomp(); icomp++) {
        CTF::Tensor<double> &dta = dca.req_ctf_tensor(icomp);
        double z = ctf_symmetry<N, double>::symconv_factor(symt, 0,
            syma, icomp);
        dta[labela] = z * dtt[labela];
        dtt[labela] -= dta[labela];
    }
}


} // namespace libtensor

#endif // CTF_TOD_DISTRIBUTE_IMPL_H

