#ifndef LIBTENSOR_TOD_TRACE_IMPL_H
#define LIBTENSOR_TOD_TRACE_IMPL_H

#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_add1.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../to_trace.h"

namespace libtensor {


template<size_t N, typename T>
const char *to_trace<N, T>::k_clazz = "to_trace<N, T>";


template<size_t N, typename T>
to_trace<N, T>::to_trace(dense_tensor_rd_i<k_ordera, T> &t) : m_t(t) {

    check_dims();
}


template<size_t N, typename T>
to_trace<N, T>::to_trace(dense_tensor_rd_i<k_ordera, T> &t,
    const permutation<k_ordera> &p) : m_t(t), m_perm(p) {

    check_dims();
}


template<size_t N, typename T>
T to_trace<N, T>::calculate() {

    T tr = 0;

    to_trace<N, T>::start_timer();

    try {

        dense_tensor_rd_ctrl<k_ordera, T> ca(m_t);
        ca.req_prefetch();

        sequence<k_ordera, size_t> map(0);
        for(register size_t i = 0; i < k_ordera; i++) map[i] = i;
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

        const T *pa = ca.req_const_dataptr();

        loop_registers_x<1, 1, T> r;
        r.m_ptra[0] = pa;
        r.m_ptrb[0] = &tr;
        r.m_ptra_end[0] = pa + dims.get_size();
        r.m_ptrb_end[0] = &tr + 1;

        {
            std::auto_ptr< kernel_base<linalg, 1, 1, T> > kern(
                    kern_add1<linalg, T>::match(1.0, loop_in, loop_out));
            to_trace<N, T>::start_timer(kern->get_name());
            loop_list_runner_x<linalg, 1, 1, T>(loop_in).run(0, r, *kern);
            to_trace<N, T>::stop_timer(kern->get_name());
        }

        ca.ret_const_dataptr(pa);

    } catch(...) {
        to_trace<N, T>::stop_timer();
        throw;
    }

    to_trace<N, T>::stop_timer();

    return tr;
}


template<size_t N, typename T>
void to_trace<N, T>::check_dims() {

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
