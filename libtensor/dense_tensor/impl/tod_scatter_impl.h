#ifndef LIBTENSOR_TOD_SCATTER_IMPL_H
#define LIBTENSOR_TOD_SCATTER_IMPL_H

#include <libtensor/tod/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_scatter.h"

namespace libtensor {


template<size_t N, size_t M>
const char *tod_scatter<N, M>::k_clazz = "tod_scatter<N, M>";


template<size_t N, size_t M>
tod_scatter<N, M>::tod_scatter(dense_tensor_rd_i<k_ordera, double> &ta,
        tensor_transf_type &trc) : m_ta(ta), m_permc(trc.get_perm()),
        m_ka(trc.get_scalar_tr().get_coeff()) {

}


template<size_t N, size_t M>
tod_scatter<N, M>::tod_scatter(dense_tensor_rd_i<k_ordera, double> &ta,
        double ka) : m_ta(ta), m_ka(ka) {

}


template<size_t N, size_t M>
tod_scatter<N, M>::tod_scatter(dense_tensor_rd_i<k_ordera, double> &ta,
        double ka, const permutation<k_orderc> &permc) :
        m_ta(ta), m_permc(permc), m_ka(ka) {

}


template<size_t N, size_t M>
void tod_scatter<N, M>::perform(bool zero,
        dense_tensor_wr_i<k_orderc, double> &tc) {

    check_dimsc(tc);

    tod_scatter<N, M>::start_timer();

    sequence<k_orderc, size_t> seq(0);
    for(size_t i = 0; i < k_orderc - k_ordera; i++) seq[i] = k_ordera;
    for(size_t i = 0; i < k_ordera; i++) seq[k_orderc - k_ordera + i] = i;
    m_permc.apply(seq);

    const dimensions<k_ordera> &dimsa = m_ta.get_dims();
    const dimensions<k_orderc> &dimsc = tc.get_dims();
    m_list.clear();
    for(size_t i = 0; i < k_orderc; i++) {
        if(seq[i] == k_ordera) {
            m_list.push_back(loop_list_node(
                dimsc[i], 0, dimsc.get_increment(i)));
        } else {
            m_list.push_back(loop_list_node(
                dimsc[i], dimsa.get_increment(seq[i]),
                dimsc.get_increment(i)));
        }
    }

    dense_tensor_rd_ctrl<k_ordera, double> ctrla(m_ta);
    dense_tensor_wr_ctrl<k_orderc, double> ctrlc(tc);

    const double *ptra = ctrla.req_const_dataptr();
    double *ptrc = ctrlc.req_dataptr();

    //  Zero the output tensor if necessary
    //
    if(zero) {
        tod_scatter<N, M>::start_timer("zero");
        size_t szc = dimsc.get_size();
        for(size_t i = 0; i < szc; i++) ptrc[i] = 0.0;
        tod_scatter<N, M>::stop_timer("zero");
    }

    //  Install the kernel on the fastest-running index in A
    //
    loop_list_iterator_t i1 = m_list.begin();
    while(i1 != m_list.end() && i1->m_inca != 1) i1++;
    if(i1 != m_list.end()) {
        i1->m_fn = &tod_scatter<N, M>::fn_scatter;
        m_scatter.m_kc = m_ka;
        m_scatter.m_n = i1->m_weight;
        m_scatter.m_stepc = i1->m_incc;
        m_list.splice(m_list.end(), m_list, i1);
    }

    //  Run the loop
    //
    try {
        registers regs;
        regs.m_ptra = ptra;
        regs.m_ptrc = ptrc;

        loop_list_iterator_t i = m_list.begin();
        if(i != m_list.end()) exec(i, regs);
    } catch(exception&) {
        tod_scatter<N, M>::stop_timer();
        throw;
    }

    ctrla.ret_const_dataptr(ptra);
    ctrlc.ret_dataptr(ptrc);

    tod_scatter<N, M>::stop_timer();
}


template<size_t N, size_t M>
void tod_scatter<N, M>::check_dimsc(dense_tensor_wr_i<k_orderc, double> &tc) {

    static const char *method =
        "check_dimsc(dense_tensor_wr_i<N + M, double>&)";

    permutation<k_orderc> pinv(m_permc, true);
    dimensions<k_orderc> dimsc(tc.get_dims());
    dimsc.permute(pinv);

    bool bad_dims = false;
    const dimensions<k_ordera> &dimsa = m_ta.get_dims();
    for(size_t i = 0; i < k_ordera; i++) {
        if(dimsc[k_orderc - k_ordera + i] != dimsa[i]) {
            bad_dims = true;
            break;
        }
    }
    if(bad_dims) {
        throw bad_dimensions(g_ns, k_clazz, method,
            __FILE__, __LINE__, "tc");
    }
}


template<size_t N, size_t M>
inline void tod_scatter<N, M>::exec(loop_list_iterator_t &i, registers &r) {

    void (tod_scatter<N, M>::*fnptr)(registers&) = i->m_fn;

    if(fnptr == 0) fn_loop(i, r);
    else (this->*fnptr)(r);
}


template<size_t N, size_t M>
void tod_scatter<N, M>::fn_loop(loop_list_iterator_t &i, registers &r) {

    loop_list_iterator_t j = i; j++;
    if(j == m_list.end()) return;

    const double *ptra = r.m_ptra;
    double *ptrc = r.m_ptrc;

    for(size_t k = 0; k < i->m_weight; k++) {

        r.m_ptra = ptra;
        r.m_ptrc = ptrc;
        exec(j, r);
        ptra += i->m_inca;
        ptrc += i->m_incc;
    }
}


template<size_t N, size_t M>
void tod_scatter<N, M>::fn_scatter(registers &r) {

    const double *ptra = r.m_ptra;
    double *ptrc = r.m_ptrc;

    for(size_t k = 0; k < m_scatter.m_n; k++) {
        ptrc[0] += m_scatter.m_kc * ptra[k];
        ptrc += m_scatter.m_stepc;
    }
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SCATTER_IMPL_H
