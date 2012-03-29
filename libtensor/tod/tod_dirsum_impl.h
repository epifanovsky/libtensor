#ifndef LIBTENSOR_TOD_DIRSUM_IMPL_H
#define LIBTENSOR_TOD_DIRSUM_IMPL_H

#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "contraction2.h"
#include "contraction2_list_builder.h"
#include "kernels/loop_list_runner.h"
#include "kernels/kern_add_generic.h"

namespace libtensor {


template<size_t N, size_t M>
const char *tod_dirsum<N, M>::k_clazz = "tod_dirsum<N, M>";


template<size_t N, size_t M>
tod_dirsum<N, M>::tod_dirsum(dense_tensor_i<k_ordera, double> &ta, double ka,
    dense_tensor_i<k_orderb, double> &tb, double kb) :

    m_ta(ta), m_tb(tb), m_ka(ka), m_kb(kb),
    m_dimsc(mk_dimsc(ta, tb)) {

}


template<size_t N, size_t M>
tod_dirsum<N, M>::tod_dirsum(dense_tensor_i<k_ordera, double> &ta, double ka,
    dense_tensor_i<k_orderb, double> &tb, double kb,
    const permutation<k_orderc> &permc) :

    m_ta(ta), m_tb(tb), m_ka(ka), m_kb(kb), m_permc(permc),
    m_dimsc(mk_dimsc(ta, tb)) {

    m_dimsc.permute(m_permc);
}


template<size_t N, size_t M>
void tod_dirsum<N, M>::perform(dense_tensor_i<k_orderc, double> &tc) {

    static const char *method = "perform(tensor_i<N + M, double>&)";

    if(!m_dimsc.equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method,
            __FILE__, __LINE__, "tc");
    }

    do_perform(tc, true, 1.0);
}


template<size_t N, size_t M>
void tod_dirsum<N, M>::perform(dense_tensor_i<k_orderc, double> &tc, double kc) {

    static const char *method =
        "perform(tensor_i<N + M, double>&, double)";

    if(!m_dimsc.equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method,
            __FILE__, __LINE__, "tc");
    }

    do_perform(tc, false, kc);
}


template<size_t N, size_t M>
dimensions<N + M> tod_dirsum<N, M>::mk_dimsc(
    dense_tensor_i<k_ordera, double> &ta, dense_tensor_i<k_orderb, double> &tb) {

    const dimensions<k_ordera> &dimsa = ta.get_dims();
    const dimensions<k_orderb> &dimsb = tb.get_dims();

    index<k_orderc> i1, i2;
    for(register size_t i = 0; i < k_ordera; i++)
        i2[i] = dimsa[i] - 1;
    for(register size_t i = 0; i < k_orderb; i++)
        i2[k_ordera + i] = dimsb[i] - 1;

    return dimensions<k_orderc>(index_range<k_orderc>(i1, i2));
}


template<size_t N, size_t M>
void tod_dirsum<N, M>::do_perform(dense_tensor_i<k_orderc, double> &tc, bool zero,
    double d) {

    tod_dirsum<N, M>::start_timer();

    try {

    dense_tensor_ctrl<k_ordera, double> ca(m_ta);
    dense_tensor_ctrl<k_orderb, double> cb(m_tb);
    dense_tensor_ctrl<k_orderc, double> cc(tc);

    ca.req_prefetch();
    cb.req_prefetch();
    cc.req_prefetch();

    const dimensions<k_ordera> &dimsa = m_ta.get_dims();
    const dimensions<k_orderb> &dimsb = m_tb.get_dims();
    const dimensions<k_orderc> &dimsc = tc.get_dims();

    std::list< loop_list_node<2, 1> > loop_in, loop_out;

    //  We use contraction2 here to build a loop list

    contraction2<N, M, 0> contr(m_permc);
    loop_list_adapter list_adapter(loop_in);
    contraction2_list_builder<N, M, 0, loop_list_adapter>(contr).
        populate(list_adapter, dimsa, dimsb, dimsc);

    const double *pa = ca.req_const_dataptr();
    const double *pb = cb.req_const_dataptr();
    double *pc = cc.req_dataptr();

    //  Zero the output tensor if necessary
    //
    if(zero) {
        tod_dirsum<N, M>::start_timer("zero");
        size_t szc = tc.get_dims().get_size();
        for(size_t i = 0; i < szc; i++) pc[i] = 0.0;
        tod_dirsum<N, M>::stop_timer("zero");
    }

    loop_registers<2, 1> r;
    r.m_ptra[0] = pa;
    r.m_ptra[1] = pb;
    r.m_ptrb[0] = pc;
    r.m_ptra_end[0] = pa + dimsa.get_size();
    r.m_ptra_end[1] = pb + dimsb.get_size();
    r.m_ptrb_end[0] = pc + dimsc.get_size();

    kernel_base<2, 1> *kern = kern_add_generic::match(m_ka, m_kb, d, loop_in,
        loop_out);
    tod_dirsum<N, M>::start_timer(kern->get_name());
    loop_list_runner<2, 1>(loop_in).run(r, *kern);
    tod_dirsum<N, M>::stop_timer(kern->get_name());
    delete kern; kern = 0;

    ca.ret_const_dataptr(pa);
    cb.ret_const_dataptr(pb);
    cc.ret_dataptr(pc);

    } catch(...) {
        tod_dirsum<N, M>::stop_timer();
        throw;
    }

    tod_dirsum<N, M>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_DIRSUM_IMPL_H
