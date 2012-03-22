#ifndef LIBTENSOR_TOD_MULT1_IMPL_H
#define LIBTENSOR_TOD_MULT1_IMPL_H

#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
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

    typedef typename loop_list_elem1::list_t list_t;
    typedef typename loop_list_elem1::registers registers_t;
    typedef typename loop_list_elem1::node node_t;

    tod_mult1<N>::start_timer();

    try {

    dense_tensor_wr_ctrl<N, double> ca(ta);
    dense_tensor_rd_ctrl<N, double> cb(m_tb);
    ca.req_prefetch();
    cb.req_prefetch();

    const dimensions<N> &dimsa = ta.get_dims();
    const dimensions<N> &dimsb = m_tb.get_dims();

    list_t loop;
    build_loop(loop, dimsa, dimsb, m_pb);

    double *pa = ca.req_dataptr();
    const double *pb = cb.req_const_dataptr();

    registers_t r;
    r.m_ptra[0] = pb;
    r.m_ptrb[0] = pa;
    r.m_ptra_end[0] = pb + dimsb.get_size();
    r.m_ptrb_end[0] = pa + dimsa.get_size();

    loop_list_elem1::run_loop(loop, r, m_c * c, doadd, m_recip);

    cb.ret_const_dataptr(pb); pb = 0;
    ca.ret_dataptr(pa); pa = 0;

    } catch (...) {
        tod_mult1<N>::stop_timer();
        throw;
    }

    tod_mult1<N>::stop_timer();

}


template<size_t N>
void tod_mult1<N>::build_loop(typename loop_list_elem1::list_t &loop,
    const dimensions<N> &dimsa, const dimensions<N> &dimsb,
    const permutation<N> &permb) {

    typedef typename loop_list_elem1::iterator_t iterator_t;
    typedef typename loop_list_elem1::node node_t;

    sequence<N, size_t> mapb(0);
    for(register size_t i = 0; i < N; i++) mapb[i] = i;
    permb.apply(mapb);

    for (size_t idxa = 0; idxa < N; ) {
        size_t len = 1;
        size_t idxb = mapb[idxa];

        do {
            len *= dimsa.get_dim(idxa);
            idxa++; idxb++;
        } while (idxa < N &&  mapb[idxa] == idxb);

        iterator_t inode = loop.insert(loop.end(), node_t(len));
        inode->stepa(0) = dimsb.get_increment(idxb - 1);
        inode->stepb(0) = dimsa.get_increment(idxa - 1);
    }

}


} // namespace libtensor

#endif // LIBTENSOR_TOD_MULT1_IMPL_H
