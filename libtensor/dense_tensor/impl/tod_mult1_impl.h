#ifndef LIBTENSOR_TOD_MULT1_IMPL_H
#define LIBTENSOR_TOD_MULT1_IMPL_H

#include <memory>
#include <libtensor/exception.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_set.h>
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
tod_mult1<N>::tod_mult1(dense_tensor_rd_i<N, double> &tb,
        const tensor_transf<N, double> &trb, bool recip,
        const scalar_transf<double> &c) :
    m_tb(tb), m_trb(trb), m_recip(recip), m_c(c)
{
    if (recip && m_trb.get_scalar_tr().get_coeff() == 0.0) {
        throw bad_parameter(g_ns, k_clazz, "tod_mult1()",
                __FILE__, __LINE__, "trb");
    }
}

template<size_t N>
void tod_mult1<N>::perform(bool zero, dense_tensor_wr_i<N, double> &ta) {

    static const char *method = "perform(dense_tensor_wr_i<N, double>&)";

    tod_mult1<N>::start_timer();

    dimensions<N> dimsb(m_tb.get_dims());
    dimsb.permute(m_trb.get_perm());

    if(!dimsb.equals(ta.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "ta");
    }

    if (m_trb.get_scalar_tr().get_coeff() == 0) {
        if (zero) tod_set<N>().perform(ta);

        return;
    }

    try {

    dense_tensor_wr_ctrl<N, double> ca(ta);
    dense_tensor_rd_ctrl<N, double> cb(m_tb);
    ca.req_prefetch();
    cb.req_prefetch();

    const dimensions<N> &dimsa = ta.get_dims();
    const dimensions<N> &dimsb = m_tb.get_dims();

    sequence<N, size_t> mapb(0);
    for(register size_t i = 0; i < N; i++) mapb[i] = i;
    m_trb.get_perm().apply(mapb);

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

    double c = (m_recip ?
            m_c.get_coeff() / m_trb.get_scalar_tr().get_coeff() :
            m_c.get_coeff() * m_trb.get_scalar_tr().get_coeff());

    loop_registers<1, 1> r;
    r.m_ptra[0] = pb;
    r.m_ptrb[0] = pa;
    r.m_ptra_end[0] = pb + dimsb.get_size();
    r.m_ptrb_end[0] = pa + dimsa.get_size();

    std::auto_ptr< kernel_base<linalg, 1, 1> > kern(
        m_recip ?
            (zero ?
                kern_ddiv1::match(c, loop_in, loop_out) :
                kern_ddivadd1::match(c, loop_in, loop_out)) :
            (zero ?
                kern_dmul1::match(c, loop_in, loop_out) :
                kern_dmuladd1::match(c, loop_in, loop_out)));
    tod_mult1<N>::start_timer(kern->get_name());
    loop_list_runner<linalg, 1, 1>(loop_in).run(0, r, *kern);
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
