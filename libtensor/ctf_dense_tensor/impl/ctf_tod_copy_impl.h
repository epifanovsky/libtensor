#ifndef LIBTENSOR_CTF_TOD_COPY_IMPL_H
#define LIBTENSOR_CTF_TOD_COPY_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_copy.h"
#include "ctf_tod_aux_symcomp.h"

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


template<size_t N>
void ctf_tod_copy<N>::perform(bool zero, ctf_dense_tensor_i<N, double> &tb) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<N, double>&)";

    if(!m_dimsb.equals(tb.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }

    ctf_dense_tensor_ctrl<N, double> ca(m_ta), cb(tb);

    double c = m_tra.get_scalar_tr().get_coeff();

    sequence<N, int> seqa(0), seqb(0);
    char labela[N + 1], labelb[N + 1];
    for(size_t i = 0; i < N; i++) seqa[i] = seqb[i] = N - i - 1;
    m_tra.get_perm().apply(seqb);
    for(size_t i = 0; i < N; i++) {
        labela[i] = seqa[N - i - 1] + 1;
        labelb[i] = seqb[N - i - 1] + 1;
    }
    labela[N] = '\0'; labelb[N] = '\0';

    const ctf_symmetry<N, double> &syma = ca.req_symmetry();
    const ctf_symmetry<N, double> &symb = cb.req_symmetry();
    ctf_symmetry<N, double> symap(syma);
    symap.permute(m_tra.get_perm());

    ctf_tod_copy::start_timer();

    std::vector<bool> zerob(symb.get_ncomp(), zero);
    for(size_t icompa = 0; icompa < symap.get_ncomp(); icompa++) {

        size_t icompb = ctf_tod_aux_symcomp(symap, icompa, symb);
        double z = ctf_symmetry<N, double>::symconv_factor(symap, icompa,
            symb, icompb);

        CTF::Tensor<double> &dta = ca.req_ctf_tensor(icompa);
        CTF::Tensor<double> &dtb = cb.req_ctf_tensor(icompb);
        if(zerob[icompb]) dtb[labelb] = c * z * dta[labela];
        else dtb[labelb] += c * z * dta[labela];
        zerob[icompb] = false;
    }
    for(size_t i = 0; i < zerob.size(); i++) if(zerob[i]) {
        CTF::Tensor<double> &dtb = cb.req_ctf_tensor(i);
        dtb = 0.0;
    }

    ctf_tod_copy::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_COPY_IMPL_H

