#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_H

#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../mp/auto_cpu_lock.h"
#include "kernels/loop_list_runner.h"
#include "kernels/kern_mul_generic.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::k_clazz = "tod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
tod_contract2<N, M, K>::tod_contract2(const contraction2<N, M, K> &contr,
    dense_tensor_i<k_ordera, double> &ta, dense_tensor_i<k_orderb, double> &tb) :

    m_contr(contr), m_ta(ta), m_tb(tb) {

}


template<size_t N, size_t M, size_t K>
tod_contract2<N, M, K>::~tod_contract2() {

}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::prefetch() {

    dense_tensor_ctrl<k_ordera, double>(m_ta).req_prefetch();
    dense_tensor_ctrl<k_orderb, double>(m_tb).req_prefetch();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::perform(cpu_pool &cpus, bool zero, double d,
    dense_tensor_i<k_orderc, double> &tc) {

    tod_contract2<N, M, K>::start_timer();

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
    loop_list_adapter list_adapter(loop_in);
    contraction2_list_builder<N, M, K, loop_list_adapter>(m_contr).
        populate(list_adapter, dimsa, dimsb, dimsc);

    const double *pa = ca.req_const_dataptr();
    const double *pb = cb.req_const_dataptr();
    double *pc = cc.req_dataptr();

    {
        auto_cpu_lock cpu(cpus);

        if(zero) {
            tod_contract2<N, M, K>::start_timer("zero");
            size_t szc = tc.get_dims().get_size();
            for(size_t i = 0; i < szc; i++) pc[i] = 0.0;
            tod_contract2<N, M, K>::stop_timer("zero");
        }

        loop_registers<2, 1> r;
        r.m_ptra[0] = pa;
        r.m_ptra[1] = pb;
        r.m_ptrb[0] = pc;
        r.m_ptra_end[0] = pa + dimsa.get_size();
        r.m_ptra_end[1] = pb + dimsb.get_size();
        r.m_ptrb_end[0] = pc + dimsc.get_size();

        kernel_base<2, 1> *kern = kern_mul_generic::match(d, loop_in, loop_out);
        tod_contract2<N, M, K>::start_timer(kern->get_name());
        loop_list_runner<2, 1>(loop_in).run(r, *kern);
        tod_contract2<N, M, K>::stop_timer(kern->get_name());
        delete kern; kern = 0;
    }

    ca.ret_const_dataptr(pa);
    cb.ret_const_dataptr(pb);
    cc.ret_dataptr(pc);

    } catch(...) {
        tod_contract2<N, M, K>::stop_timer();
        throw;
    }

    tod_contract2<N, M, K>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_H
