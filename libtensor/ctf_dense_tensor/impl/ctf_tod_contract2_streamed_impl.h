#ifndef LIBTENSOR_CTF_TOD_CONTRACT2_STREAMED_IMPL_H
#define LIBTENSOR_CTF_TOD_CONTRACT2_STREAMED_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/dense_tensor/to_contract2_dims.h>
#include "../ctf_tod_contract2.h"
#include "../ctf_tod_contract2_streamed.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char ctf_tod_contract2_streamed<N, M, K>::k_clazz[] =
    "ctf_tod_contract2_streamed<N, M, K>";


template<size_t N, size_t M, size_t K>
ctf_tod_contract2_streamed<N, M, K>::ctf_tod_contract2_streamed(
    const contraction2<N, M, K> &contr,
    ctf_dense_tensor_i<NA, double> &ta,
    const scalar_transf<double> &ka,
    ctf_dense_tensor_i<NB, double> &tb,
    const scalar_transf<double> &kb,
    const scalar_transf<double> &kc) :

    m_dimsc(to_contract2_dims<N, M, K>(contr, ta.get_dims(), tb.get_dims()).
        get_dims()) {

    add_args(contr, ta, ka, tb, kb, kc);
}


template<size_t N, size_t M, size_t K>
void ctf_tod_contract2_streamed<N, M, K>::add_args(
    const contraction2<N, M, K> &contr,
    ctf_dense_tensor_i<NA, double> &ta,
    const scalar_transf<double> &ka,
    ctf_dense_tensor_i<NB, double> &tb,
    const scalar_transf<double> &kb,
    const scalar_transf<double> &kc) {

    static const char method[] = "add_args()";

    if(!to_contract2_dims<N, M, K>(contr, ta.get_dims(), tb.get_dims()).
        get_dims().equals(m_dimsc)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta,tb");
    }

    double d = ka.get_coeff() * kb.get_coeff() * kc.get_coeff();
    m_argslst.push_back(args(contr, ta, tb, d));
}


template<size_t N, size_t M, size_t K>
void ctf_tod_contract2_streamed<N, M, K>::perform(
    bool zero,
    ctf_dense_tensor_i<NC, double> &tc) {

    bool z = zero;
    for(typename std::list<args>::iterator i = m_argslst.begin();
        i != m_argslst.end(); ++i) {

        ctf_tod_contract2<N, M, K>(i->contr, i->ta, i->tb, i->d).perform(z, tc);
        z = false;
    }
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_CONTRACT2_STREAMED_IMPL_H

