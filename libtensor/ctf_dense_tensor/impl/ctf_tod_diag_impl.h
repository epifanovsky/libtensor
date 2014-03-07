#ifndef LIBTENSOR_CTF_TOD_DIAG_IMPL_H
#define LIBTENSOR_CTF_TOD_DIAG_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/dense_tensor/to_diag_dims.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_diag.h"

namespace libtensor {


template<size_t N, size_t M>
const char ctf_tod_diag<N, M>::k_clazz[] = "ctf_tod_diag<N, M>";


template<size_t N, size_t M>
ctf_tod_diag<N, M>::ctf_tod_diag(
    ctf_dense_tensor_i<NA, double> &ta,
    const mask<NA> &m,
    const tensor_transf<NB, double> &trb) :

    m_ta(ta), m_mask(m), m_trb(trb),
    m_dimsb(to_diag_dims<N, M>(ta.get_dims(), m, trb.get_perm()).get_dimsb()) {

}


template<size_t N, size_t M>
void ctf_tod_diag<N, M>::perform(
    bool zero,
    ctf_dense_tensor_i<NB, double> &tb) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<NB, double>&)";

    if(!m_dimsb.equals(tb.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }

    ctf_dense_tensor_ctrl<NA, double> ca(m_ta);
    ctf_dense_tensor_ctrl<NB, double> cb(tb);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();
    tCTF_Tensor<double> &dtb = cb.req_ctf_tensor();

    double c = m_trb.get_scalar_tr().get_coeff();

    sequence<NA, int> seqa(0);
    sequence<NB, int> seqb(0);
    int didx = NA;
    bool first = true;
    for(size_t i = 0, j = 0; i < NA; i++) {
        if(m_mask[i]) {
            seqa[i] = didx;
            if(first) {
                seqb[j++] = didx;
                first = false;
            }
        } else {
            seqb[j++] = seqa[i] = int(i);
        }
    }
    m_trb.get_perm().apply(seqb);

    char mapa[NA], mapb[NB];
    for(size_t i = 0; i < NA; i++) mapa[i] = seqa[NA - i - 1] + 1;
    for(size_t i = 0; i < NB; i++) mapb[i] = seqb[NB - i - 1] + 1;

    dtb.sum(c, dta, mapa, zero ? 0.0 : 1.0, mapb);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_DIAG_IMPL_H

