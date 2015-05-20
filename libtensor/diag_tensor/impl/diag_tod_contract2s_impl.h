#ifndef LIBTENSOR_DIAG_TOD_CONTRACT2S_IMPL_H
#define LIBTENSOR_DIAG_TOD_CONTRACT2S_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../diag_tod_contract2.h"
#include "../diag_tod_set.h"
#include "../diag_tod_contract2s.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *diag_tod_contract2s<N, M, K>::k_clazz =
    "diag_tod_contract2s<N, M, K>";


template<size_t N, size_t M, size_t K>
diag_tod_contract2s<N, M, K>::diag_tod_contract2s(
    const contraction2<N, M, K> &contr,
    diag_tensor_rd_i<NA, double> &dta,
    const scalar_transf<double> &ka,
    diag_tensor_rd_i<NB, double> &dtb,
    const scalar_transf<double> &kb,
    const scalar_transf<double> &kc) :

    m_dimsc(contr, dta.get_dims(), dtb.get_dims()) {

    add_args(contr, dta, ka, dtb, kb, kc);
}


template<size_t N, size_t M, size_t K>
diag_tod_contract2s<N, M, K>::diag_tod_contract2s(
    const contraction2<N, M, K> &contr,
    diag_tensor_rd_i<NA, double> &dta,
    diag_tensor_rd_i<NB, double> &dtb,
    double d) :

    m_dimsc(contr, dta.get_dims(), dtb.get_dims()) {

    add_args(contr, dta, dtb, d);
}


template<size_t N, size_t M, size_t K>
void diag_tod_contract2s<N, M, K>::add_args(
    const contraction2<N, M, K> &contr,
    diag_tensor_rd_i<NA, double> &dta,
    const scalar_transf<double> &ka,
    diag_tensor_rd_i<NB, double> &dtb,
    const scalar_transf<double> &kb,
    const scalar_transf<double> &kc) {

    double d = ka.get_coeff() * kb.get_coeff() * kc.get_coeff();
    add_args(contr, dta, dtb, d);
}


template<size_t N, size_t M, size_t K>
void diag_tod_contract2s<N, M, K>::add_args(
    const contraction2<N, M, K> &contr,
    diag_tensor_rd_i<NA, double> &dta,
    diag_tensor_rd_i<NB, double> &dtb,
    double d) {

    static const char *method = "add_args(const contraction2<N, M, K>&, "
        "dense_tensor_i<N + K, double>&, dense_tensor_i<M + K, double>&, "
        "double)";

    if(!to_contract2_dims<N, M, K>(contr, dta.get_dims(), dtb.get_dims()).
        get_dims().equals(m_dimsc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "dta,dtb");
    }

    m_argslst.push_back(args(contr, dta, dtb, d));
}


template<size_t N, size_t M, size_t K>
void diag_tod_contract2s<N, M, K>::perform(
    bool zero, diag_tensor_wr_i<N + M, double> &dtc) {

    bool zero1 = zero;
    for(typename std::list<args>::iterator i = m_argslst.begin();
        i != m_argslst.end(); ++i) {

        diag_tod_contract2<N, M, K>(i->contr, i->dta, i->dtb, i->d).perform(
            zero1, dtc);
        zero1 = false;
    }
    if(zero1) {
        diag_tod_set<NC>().perform(zero1, dtc);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_CONTRACT2S_IMPL_H
