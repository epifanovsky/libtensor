#ifndef LIBTENSOR_TOD_TRACE_IMPL_H
#define LIBTENSOR_TOD_TRACE_IMPL_H

#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dadd1.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
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

    double tr = 0;

    tod_trace<N>::start_timer();

    try {

        dense_tensor_rd_ctrl<k_ordera, double> ca(m_t);
        ca.req_prefetch();

        sequence<k_ordera, size_t> map(0);
        for(size_t i = 0; i < k_ordera; i++) map[i] = i;
        permutation<k_ordera> pinv(m_perm, true);
        pinv.apply(map);

        std::list< loop_list_node<1, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<1, 1> >::iterator inode =
                loop_in.end();

        const dimensions<k_ordera> &dims = m_t.get_dims();
        for(size_t i = 0; i < N; i++) {
            inode = loop_in.insert(loop_in.end(),
                    loop_list_node<1, 1>(dims[map[i]]));
            inode->stepa(0) = dims.get_increment(map[i]) +
                    dims.get_increment(map[N + i]);
            inode->stepb(0) = 0;
        }

        const double *pa = ca.req_const_dataptr();

        loop_registers<1, 1> r;
        r.m_ptra[0] = pa;
        r.m_ptrb[0] = &tr;
        r.m_ptra_end[0] = pa + dims.get_size();
        r.m_ptrb_end[0] = &tr + 1;

        {
            std::auto_ptr< kernel_base<linalg, 1, 1> > kern(
                    kern_dadd1<linalg>::match(1.0, loop_in, loop_out));
            tod_trace<N>::start_timer(kern->get_name());
            loop_list_runner<linalg, 1, 1>(loop_in).run(0, r, *kern);
            tod_trace<N>::stop_timer(kern->get_name());
        }

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
    for(size_t i = 0; i < k_ordera; i++) map[i] = i;
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
