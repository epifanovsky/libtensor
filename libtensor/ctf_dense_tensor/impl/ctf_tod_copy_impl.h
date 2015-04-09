#ifndef LIBTENSOR_CTF_TOD_COPY_IMPL_H
#define LIBTENSOR_CTF_TOD_COPY_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_copy.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_copy<N>::k_clazz[] = "ctf_tod_copy<N>";


template<size_t N>
ctf_tod_copy<N>::ctf_tod_copy(ctf_dense_tensor_i<N, double> &ta, double c) :

    m_ta(ta), m_tra(permutation<N>(), scalar_transf<double>(c)),
    m_dimsb(m_ta.get_dims()) {

}


template<size_t N>
ctf_tod_copy<N>::ctf_tod_copy(ctf_dense_tensor_i<N, double> &ta,
    const permutation<N> &perma, double c) :

    m_ta(ta), m_tra(perma, scalar_transf<double>(c)),
    m_dimsb(m_ta.get_dims()) {

    m_dimsb.permute(perma);
}


template<size_t N>
ctf_tod_copy<N>::ctf_tod_copy(ctf_dense_tensor_i<N, double> &ta,
    const tensor_transf<N, double> &tra) :

    m_ta(ta), m_tra(tra), m_dimsb(m_ta.get_dims()) {

    m_dimsb.permute(m_tra.get_perm());
}


namespace {

unsigned long symmetry_factor(size_t N, int *sym) {

    unsigned long f = 1;
    for(size_t i = 0, j = 0; i < N; i++) {
        if(sym[i] == SY || sym[i] == AS) {
            j++;
            continue;
        }
        if(sym[i] == NS) {
            j++;
            for(size_t k = 2; k <= j; k++) f *= k;
            j = 0;
        }
    }
    return f;
}

} // unnamed namespace


template<size_t N>
void ctf_tod_copy<N>::perform(bool zero, ctf_dense_tensor_i<N, double> &tb) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<N, double>&)";

    if(!m_dimsb.equals(tb.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }

    ctf_dense_tensor_ctrl<N, double> ca(m_ta), cb(tb);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();
    tCTF_Tensor<double> &dtb = cb.req_ctf_tensor();

    double c = m_tra.get_scalar_tr().get_coeff();

    sequence<N, int> seqa(0), seqb(0);
    char mapa[N], mapb[N];
    for(size_t i = 0; i < N; i++) seqa[i] = seqb[i] = N - i - 1;
    m_tra.get_perm().apply(seqb);
    for(size_t i = 0; i < N; i++) {
        mapa[i] = seqa[N - i - 1] + 1;
        mapb[i] = seqb[N - i - 1] + 1;
    }

    //  Tricky part: CTF implies symmetrization when the symmetry of B is
    //  higher than the symmetry of A. Need to correct with an appropriate
    //  factor for bug-free computing
    sequence<N, int> syma, symb;
    for(size_t i = 0; i < N; i++) {
        syma[i] = dta.sym[i];
        symb[i] = dtb.sym[i];
    }
    for(size_t i = 0; i < N; i++) {
        if(symb[i] == NS) syma[i] = NS;
    }
    unsigned long fa = symmetry_factor(N, &syma[0]);
    unsigned long fb = symmetry_factor(N, &symb[0]);

    dtb.sum(c / double(fb / fa), dta, mapa, zero ? 0.0 : 1.0, mapb);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_COPY_IMPL_H

