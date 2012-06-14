#ifndef LIBTENSOR_TOD_DIRSUM_IMPL_H
#define LIBTENSOR_TOD_DIRSUM_IMPL_H

#include <memory>
#include <libtensor/tod/bad_dimensions.h>
#include <libtensor/tod/contraction2.h>
#include <libtensor/tod/contraction2_list_builder.h>
#include <libtensor/tod/kernels/loop_list_runner.h>
#include <libtensor/tod/kernels/kern_add_generic.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_dirsum.h"

namespace libtensor {


template<size_t N, size_t M>
const char *tod_dirsum<N, M>::k_clazz = "tod_dirsum<N, M>";


template<size_t N, size_t M>
tod_dirsum<N, M>::tod_dirsum(dense_tensor_rd_i<k_ordera, double> &ta, double ka,
    dense_tensor_rd_i<k_orderb, double> &tb, double kb) :

    m_ta(ta), m_tb(tb), m_ka(ka), m_kb(kb), m_dimsc(mk_dimsc(ta, tb)) {

}


template<size_t N, size_t M>
tod_dirsum<N, M>::tod_dirsum(dense_tensor_rd_i<k_ordera, double> &ta, double ka,
    dense_tensor_rd_i<k_orderb, double> &tb, double kb,
    const permutation<k_orderc> &permc) :

    m_ta(ta), m_tb(tb), m_ka(ka), m_kb(kb), m_permc(permc),
    m_dimsc(mk_dimsc(ta, tb)) {

    m_dimsc.permute(m_permc);
}


template<size_t N, size_t M>
void tod_dirsum<N, M>::perform(bool zero, double d,
    dense_tensor_wr_i<k_orderc, double> &tc) {

    static const char *method = "perform(bool, double, "
        "dense_tensor_wr_i<N + M, double>&)";

    if(!m_dimsc.equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method,
            __FILE__, __LINE__, "tc");
    }

    tod_dirsum<N, M>::start_timer();

    try {

        dense_tensor_rd_ctrl<k_ordera, double> ca(m_ta);
        dense_tensor_rd_ctrl<k_orderb, double> cb(m_tb);
        dense_tensor_wr_ctrl<k_orderc, double> cc(tc);

        ca.req_prefetch();
        cb.req_prefetch();
        cc.req_prefetch();

        sequence<k_orderc, size_t> seqc(0);
        for(size_t i = 0; i < k_orderc; i++) seqc[i] = i;
        m_permc.apply(seqc);

        const dimensions<k_ordera> &dimsa = m_ta.get_dims();
        const dimensions<k_orderb> &dimsb = m_tb.get_dims();
        const dimensions<k_orderc> &dimsc = tc.get_dims();

        std::list< loop_list_node<2, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<2, 1> >::iterator inode =
            loop_in.end();
        for(size_t i = 0; i < k_orderc; i++) {
            size_t ic = seqc[i];
            size_t w, inca, incb;
            if(ic < N) {
                size_t ia = ic;
                w = dimsa[ia];
                inca = dimsa.get_increment(ia);
                incb = 0;
            } else {
                size_t ib = ic - N;
                w = dimsb[ib];
                inca = 0;
                incb = dimsb.get_increment(ib);
            }
            inode = loop_in.insert(loop_in.end(), loop_list_node<2, 1>(w));
            inode->stepa(0) = inca;
            inode->stepa(1) = incb;
            inode->stepb(0) = dimsc.get_increment(i);
        }

        const double *pa = ca.req_const_dataptr();
        const double *pb = cb.req_const_dataptr();
        double *pc = cc.req_dataptr();

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

        {
            std::auto_ptr< kernel_base<2, 1> >kern(
                kern_add_generic::match(m_ka, m_kb, d, loop_in, loop_out));
            tod_dirsum<N, M>::start_timer(kern->get_name());
            loop_list_runner<2, 1>(loop_in).run(r, *kern);
            tod_dirsum<N, M>::stop_timer(kern->get_name());
        }

        ca.ret_const_dataptr(pa); pa = 0;
        cb.ret_const_dataptr(pb); pb = 0;
        cc.ret_dataptr(pc); pc = 0;

    } catch(...) {
        tod_dirsum<N, M>::stop_timer();
        throw;
    }

    tod_dirsum<N, M>::stop_timer();
}


template<size_t N, size_t M>
dimensions<N + M> tod_dirsum<N, M>::mk_dimsc(
    dense_tensor_rd_i<k_ordera, double> &ta,
    dense_tensor_rd_i<k_orderb, double> &tb) {

    const dimensions<k_ordera> &dimsa = ta.get_dims();
    const dimensions<k_orderb> &dimsb = tb.get_dims();

    index<k_orderc> i1, i2;
    for(size_t i = 0; i < k_ordera; i++) i2[i] = dimsa[i] - 1;
    for(size_t i = 0; i < k_orderb; i++) i2[k_ordera + i] = dimsb[i] - 1;

    return dimensions<k_orderc>(index_range<k_orderc>(i1, i2));
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_DIRSUM_IMPL_H
