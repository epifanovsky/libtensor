#ifndef LIBTENSOR_CTF_TOD_TRACE_IMPL_H
#define LIBTENSOR_CTF_TOD_TRACE_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_tod_trace.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_trace<N>::k_clazz[] = "ctf_tod_trace<N>";


template<size_t N>
ctf_tod_trace<N>::ctf_tod_trace(ctf_dense_tensor_i<NA, double> &ta) :

    m_ta(ta) {

    check_dims();
}


template<size_t N>
ctf_tod_trace<N>::ctf_tod_trace(
    ctf_dense_tensor_i<NA, double> &ta,
    const permutation<NA> &perma) :

    m_ta(ta), m_perma(perma) {

    check_dims();
}


template<size_t N>
double ctf_tod_trace<N>::calculate() {

    ctf_dense_tensor_ctrl<NA, double> ca(m_ta);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();

    sequence<NA, int> seqa(0);
    sequence<N, int> seqb(0);
    for(size_t i = 0; i < N; i++) seqb[i] = seqa[N + i] = seqa[i] = int(i);
    permutation<NA>(m_perma, true).apply(seqa);

    char mapa[NA], mapb[N];
    for(size_t i = 0; i < NA; i++) mapa[i] = seqa[NA - i - 1] + 1;
    for(size_t i = 0; i < N; i++) mapb[i] = seqb[N - i - 1] + 1;

    int dimsb[N], symb[N];
    dimensions<NA> dimsa1(m_ta.get_dims());
    dimsa1.permute(m_perma);
    for(size_t i = 0; i < N; i++) {
        dimsb[i] = int(dimsa1[N - i - 1]);
        symb[i] = 0;
    }
    tCTF_Tensor<double> dtb(N, dimsb, symb, ctf::get_world());
std::cout << "A ["; for(int i = 0; i < NA; i++) std::cout << dta.len[i] << " "; std::cout << "]; ";
std::cout << "B ["; for(int i = 0; i < N; i++) std::cout << dtb.len[i] << " "; std::cout << "]" << std::endl;
std::cout << "("; for(int i = 0; i < N; i++) std::cout << int(mapb[i]); std::cout << ") <- "; 
std::cout << "("; for(int i = 0; i < NA; i++) std::cout << int(mapa[i]); std::cout << ")" << std::endl;
    dtb.sum(1.0, dta, mapa, 0.0, mapb);
    return dtb.reduce(CTF_OP_SUM);
}


template<size_t N>
void ctf_tod_trace<N>::check_dims() {

    static const char method[] = "check_dims()";

    sequence<NA, size_t> map(0);
    for(size_t i = 0; i < NA; i++) map[i] = i;
    permutation<NA> pinv(m_perma, true);
    pinv.apply(map);

    const dimensions<NA> &dims = m_ta.get_dims();
    for(size_t i = 0; i < N; i++) {
        if(dims[map[i]] != dims[map[N + i]]) {
            throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
                "t");
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_TRACE_IMPL_H
