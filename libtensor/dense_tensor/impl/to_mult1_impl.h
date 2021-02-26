#ifndef LIBTENSOR_TO_MULT1_IMPL_H
#define LIBTENSOR_TO_MULT1_IMPL_H

#include <memory>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_div1.h>
#include <libtensor/kernels/kern_divadd1.h>
#include <libtensor/kernels/kern_mul1.h>
#include <libtensor/kernels/kern_muladd1.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../to_mult1.h"
#include "../to_set.h"

namespace libtensor {


template<size_t N, typename T>
const char *to_mult1<N, T>::k_clazz = "to_mult1<N, T>";

template<size_t N, typename T>
to_mult1<N, T>::to_mult1(dense_tensor_rd_i<N, T> &tb,
        const tensor_transf<N, T> &trb, bool recip,
        const scalar_transf<T> &c) :
    m_tb(tb), m_permb(trb.get_perm()), m_recip(recip), m_c(c.get_coeff())
{
    if (recip && trb.get_scalar_tr().get_coeff() == 0.0) {
        throw bad_parameter(g_ns, k_clazz, "to_mult1()",
                __FILE__, __LINE__, "trb");
    }

    m_c = (recip ?
            m_c / trb.get_scalar_tr().get_coeff() :
            m_c * trb.get_scalar_tr().get_coeff());
}

template<size_t N, typename T>
void to_mult1<N, T>::perform(bool zero, dense_tensor_wr_i<N, T> &ta) {

    static const char *method = "perform(dense_tensor_wr_i<N, T>&)";

    to_mult1<N, T>::start_timer();

    dimensions<N> dimsb(m_tb.get_dims());
    dimsb.permute(m_permb);

    if(!dimsb.equals(ta.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "ta");
    }

    if (m_c == 0) {
        if (zero) to_set<N, T>().perform(zero, ta);
        return;
    }

    try {

    dense_tensor_wr_ctrl<N, T> ca(ta);
    dense_tensor_rd_ctrl<N, T> cb(m_tb);
    ca.req_prefetch();
    cb.req_prefetch();

    const dimensions<N> &dimsa = ta.get_dims();
    const dimensions<N> &dimsb = m_tb.get_dims();

    sequence<N, size_t> mapb(0);
    for(size_t i = 0; i < N; i++) mapb[i] = i;
    m_permb.apply(mapb);

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

    T *pa = ca.req_dataptr();
    const T *pb = cb.req_const_dataptr();

    loop_registers_x<1, 1, T> r;
    r.m_ptra[0] = pb;
    r.m_ptrb[0] = pa;
    r.m_ptra_end[0] = pb + dimsb.get_size();
    r.m_ptrb_end[0] = pa + dimsa.get_size();

    std::unique_ptr< kernel_base<linalg, 1, 1, T> > kern(
        m_recip ?
            (zero ?
                kern_div1<linalg, T>::match(m_c, loop_in, loop_out) :
                kern_divadd1<T>::match(m_c, loop_in, loop_out)) :
            (zero ?
                kern_mul1<T>::match(m_c, loop_in, loop_out) :
                kern_muladd1<T>::match(m_c, loop_in, loop_out)));
    to_mult1<N, T>::start_timer(kern->get_name());
    loop_list_runner_x<linalg, 1, 1, T>(loop_in).run(0, r, *kern);
    to_mult1<N, T>::stop_timer(kern->get_name());

    cb.ret_const_dataptr(pb); pb = 0;
    ca.ret_dataptr(pa); pa = 0;

    } catch (...) {
        to_mult1<N, T>::stop_timer();
        throw;
    }

    to_mult1<N, T>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TO_MULT1_IMPL_H
