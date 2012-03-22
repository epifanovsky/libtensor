#ifndef LIBTENSOR_TOD_TRACE_IMPL_H
#define LIBTENSOR_TOD_TRACE_IMPL_H

#include <libtensor/linalg.h>
#include <libtensor/tod/bad_dimensions.h>
#include "../tod_trace.h"

namespace libtensor {


template<size_t N>
const char *tod_trace<N>::k_clazz = "tod_trace<N>";


template<size_t N>
tod_trace<N>::tod_trace(dense_tensor_rd_i<k_ordera, double> &t) : m_t(t) {

    check_dims();
}


template<size_t N>
tod_trace<N>::tod_trace(dense_tensor_rd_i<k_ordera, double> &t,
    const permutation<k_ordera> &p) : m_t(t), m_perm(p) {

    check_dims();
}


template<size_t N>
double tod_trace<N>::calculate() {

    typedef typename loop_list_add::registers registers_t;
    typedef typename loop_list_add::node node_t;
    typedef typename loop_list_add::list_t list_t;
    typedef typename loop_list_add::iterator_t iterator_t;

    double tr = 0;

    try {

    tod_trace<N>::start_timer();

    dense_tensor_rd_ctrl<k_ordera, double> ca(m_t);
    ca.req_prefetch();

    sequence<k_ordera, size_t> map(0);
    for(register size_t i = 0; i < k_ordera; i++) map[i] = i;
    permutation<k_ordera> pinv(m_perm, true);
    pinv.apply(map);

    list_t loop;

    const dimensions<k_ordera> &dims = m_t.get_dims();
    for(size_t i = 0; i < N; i++) {
        size_t weight = dims[map[i]];
        size_t inc = dims.get_increment(map[i]) +
            dims.get_increment(map[N + i]);
        iterator_t inode = loop.insert(loop.end(), node_t(weight));
        inode->stepa(0) = inc;
        inode->stepb(0) = 0;
    }

    const double *pa = ca.req_const_dataptr();

    registers_t r;
    r.m_ptra[0] = pa;
    r.m_ptrb[0] = &tr;
#ifdef LIBTENSOR_DEBUG
    r.m_ptra_end[0] = pa + dims.get_size();
    r.m_ptrb_end[0] = &tr + 1;
#endif // LIBTENSOR_DEBUG

    loop_list_add::run_loop(loop, r, 1.0);

    ca.ret_const_dataptr(pa);

    } catch(...) {
        tod_trace<N>::stop_timer();
        throw;
    }

    tod_trace<N>::stop_timer();

    return tr;
}


template<size_t N>
void tod_trace<N>::check_dims() {

    static const char *method = "check_dims()";

    sequence<k_ordera, size_t> map(0);
    for(register size_t i = 0; i < k_ordera; i++) map[i] = i;
    permutation<k_ordera> pinv(m_perm, true);
    pinv.apply(map);

    const dimensions<k_ordera> &dims = m_t.get_dims();
    for(size_t i = 0; i < N; i++) {
        if(dims[map[i]] != dims[map[N + i]]) {
            throw bad_dimensions(g_ns, k_clazz, method,
                __FILE__, __LINE__, "t");
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_TRACE_IMPL_H
