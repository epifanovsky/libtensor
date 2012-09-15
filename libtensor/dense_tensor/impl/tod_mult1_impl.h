#ifndef LIBTENSOR_TOD_MULT1_IMPL_H
#define LIBTENSOR_TOD_MULT1_IMPL_H

#include <memory>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/kernels/kern_ddiv1.h>
#include <libtensor/kernels/kern_ddivadd1.h>
#include <libtensor/kernels/kern_dmul1.h>
#include <libtensor/kernels/kern_dmuladd1.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/tod/bad_dimensions.h>
#include "../tod_mult1.h"

namespace libtensor {


template<size_t N>
const char *tod_mult1<N>::k_clazz = "tod_mult1<N>";


template<size_t N>
void tod_mult1<N>::perform(dense_tensor_wr_i<N, double> &ta) {

    static const char *method = "perform(dense_tensor_wr_i<N, double>&)";

    dimensions<N> dimsb(m_tb.get_dims());
    dimsb.permute(m_pb);

    if(!dimsb.equals(ta.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "ta");
    }

    do_perform(ta, false, 1.0);

}


template<size_t N>
void tod_mult1<N>::perform(dense_tensor_wr_i<N, double> &ta, double c) {

    static const char *method =
        "perform(dense_tensor_wr_i<N, double>&, double)";

    dimensions<N> dimsb(m_tb.get_dims());
    dimsb.permute(m_pb);

    if(!dimsb.equals(ta.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "ta");
    }

    do_perform(ta, true, c);
}


template<size_t N>
void tod_mult1<N>::do_perform(dense_tensor_wr_i<N, double> &ta, bool doadd,
    double c) {

    tod_mult1<N>::start_timer();

    try {

    dense_tensor_wr_ctrl<N, double> ca(ta);
    dense_tensor_rd_ctrl<N, double> cb(m_tb);
    ca.req_prefetch();
    cb.req_prefetch();

    const dimensions<N> &dimsa = ta.get_dims();
    const dimensions<N> &dimsb = m_tb.get_dims();

    sequence<N, size_t> mapb(0);
    for(register size_t i = 0; i < N; i++) mapb[i] = i;
    m_pb.apply(mapb);

    std::list< loop_list_node<1, 1> > loop_in, loop_out;
    typename std::list< loop_list_node<1, 1> >::iterator inode = loop_in.end();
    for (size_t idxa = 0; idxa < N; ) {
        size_t len = 1;
        size_t idxb = mapb[idxa];

        do {
            len *= dimsa.get_dim(idxa);
            idxa++; idxb++;
        } while (idxa < N &&  mapb[idxa] == idxb);

        inode = loop_in.insert(loop_in.end(), loop_list_node<1, 1>(len));
        inode->stepa(0) = dimsb.get_increment(idxb - 1);
        inode->stepb(0) = dimsa.get_increment(idxa - 1);
    }

    double *pa = ca.req_dataptr();
    const double *pb = cb.req_const_dataptr();

    loop_registers<1, 1> r;
    r.m_ptra[0] = pb;
    r.m_ptrb[0] = pa;
    r.m_ptra_end[0] = pb + dimsb.get_size();
    r.m_ptrb_end[0] = pa + dimsa.get_size();

    std::auto_ptr< kernel_base<1, 1> > kern(
        m_recip ?
            (doadd ?
                kern_ddivadd1::match(m_c * c, loop_in, loop_out) :
                kern_ddiv1::match(m_c * c, loop_in, loop_out)) :
            (doadd ?
                kern_dmuladd1::match(m_c * c, loop_in, loop_out) :
                kern_dmul1::match(m_c * c, loop_in, loop_out)));
    tod_mult1<N>::start_timer(kern->get_name());
    loop_list_runner<1, 1>(loop_in).run(r, *kern);
    tod_mult1<N>::stop_timer(kern->get_name());

    cb.ret_const_dataptr(pb); pb = 0;
    ca.ret_dataptr(pa); pa = 0;

    } catch (...) {
        tod_mult1<N>::stop_timer();
        throw;
    }

    tod_mult1<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_MULT1_IMPL_H
